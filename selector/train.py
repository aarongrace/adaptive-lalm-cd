"""
Selector training with the paper's reported hyperparameters (Section VI-A and
VI-D): AdamW with a short warmup and cosine decay, multi-label BCE against the
oracle's multi-hot target, and light regularisation.

Training optimises ordinary BCE over all candidate branches at once, treating
every branch that answered an example correctly as a positive. At inference
the selector routes to whichever candidate has the highest predicted logit,
and is scored by whether that branch is oracle-correct -- which is exactly the
accuracy the selector would deliver if its choice were used as the negative
branch for contrastive decoding.

Why regularisation matters here more than usual: the selector sees roughly
7,500 training examples against an input of several thousand dimensions, so
training and validation diverge within a couple of dozen epochs if left alone.
The paper's unregularised baseline peaks at 75.6% with early stopping firing
around epoch 25; the reported configuration (label smoothing 0.25, feature
noise 0.10, input dropout 0.05) extends the useful horizon to roughly 75
epochs and reaches 76.7%.

Label smoothing is the single most effective ingredient, and for a specific
reason: the binary oracle targets are genuinely noisy. Whether branch A beats
branch B on one finite example reflects sampling variance as much as any true
utility ordering, so forcing the classifier to commit to hard 0/1 targets
makes it memorise that noise. Softening the targets lets it keep refining its
decision boundary instead. The other knobs below (mixup, feature dropout) were
explored and are kept configurable, but are off in the reported optimum.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

Inputs = tuple[torch.Tensor, ...]
Batch = tuple[Inputs, torch.Tensor]


@dataclass(frozen=True)
class TrainConfig:
    """Optimiser and regularisation settings; defaults are the reported optimum."""

    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 140
    patience: int = 20
    warmup_epochs: int = 3
    min_lr_ratio: float = 0.01
    grad_clip: float = 1.0
    batch_size: int = 1024
    # --- regularisation (paper Section VI-D) ---
    label_smoothing: float = 0.25
    feature_noise_std: float = 0.10
    input_dropout: float = 0.05
    feature_dropout: float = 0.0   # explored; not part of the reported optimum
    mixup_alpha: float = 0.0       # explored; not part of the reported optimum

    @classmethod
    def unregularized(cls) -> "TrainConfig":
        """The Section VI-D baseline: same optimiser, every regulariser off."""
        return cls(label_smoothing=0.0, feature_noise_std=0.0, input_dropout=0.0,
                   feature_dropout=0.0, mixup_alpha=0.0, weight_decay=0.0)

    def with_overrides(self, **kwargs) -> "TrainConfig":
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})


def _smooth(labels: torch.Tensor, eps: float) -> torch.Tensor:
    """Pull multi-hot targets off the 0/1 endpoints toward 0.5."""
    return labels * (1.0 - eps) + 0.5 * eps if eps > 0 else labels


def _augment(inputs: Inputs, cfg: TrainConfig) -> Inputs:
    """Perturb the pooled feature vector during training.

    Only the first input tensor is touched. For a cross-attention feature the
    remaining tensors are the key/value frames and their boolean mask, where
    additive noise or dropout would be meaningless (and, on the mask, wrong).
    """
    x = inputs[0]
    if cfg.feature_noise_std > 0:
        x = x + torch.randn_like(x) * cfg.feature_noise_std
    if cfg.input_dropout > 0:
        x = x * (torch.rand_like(x) > cfg.input_dropout).float()
    if cfg.feature_dropout > 0:
        # Drops whole feature dimensions for the entire batch, rather than
        # per-element, so the head cannot lean on any single coordinate.
        keep = (torch.rand(x.shape[-1], device=x.device) > cfg.feature_dropout).float()
        x = x * keep
    return (x,) + tuple(inputs[1:])


def _mixup(inputs: Inputs, labels: torch.Tensor, alpha: float) -> tuple[Inputs, torch.Tensor]:
    """Convex-combine pairs of pooled features and their targets."""
    if alpha <= 0:
        return inputs, labels
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(labels.shape[0], device=labels.device)
    mixed = lam * inputs[0] + (1.0 - lam) * inputs[0][perm]
    mixed_labels = lam * labels + (1.0 - lam) * labels[perm]
    return (mixed,) + tuple(inputs[1:]), mixed_labels


def top1_hit_rate(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of examples whose argmax-routed branch is oracle-correct."""
    top1 = logits.argmax(dim=1)
    rows = torch.arange(len(top1), device=labels.device)
    return labels[rows, top1].float().mean().item()


def _loader(data: Batch, cfg: TrainConfig, shuffle: bool) -> DataLoader:
    inputs, labels = data
    return DataLoader(TensorDataset(*inputs, labels), batch_size=cfg.batch_size, shuffle=shuffle)


def _to_device(batch: Sequence[torch.Tensor], device: str) -> Batch:
    moved = [t.to(device) for t in batch]
    return tuple(moved[:-1]), moved[-1]


def train_selector(model: nn.Module, train_data: Batch, val_data: Batch,
                   cfg: TrainConfig = TrainConfig(),
                   device: Optional[str] = None) -> tuple[nn.Module, dict]:
    """Train one selector head; returns it with its best-validation weights loaded.

    Model selection is on validation *loss* rather than validation routing
    accuracy. Accuracy over a handful of candidate branches is a coarse, jumpy
    signal on a validation set this size, so selecting on it would mostly pick
    a lucky epoch; the loss moves smoothly and generalises better to the
    held-out test partition.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.epochs - cfg.warmup_epochs), eta_min=cfg.lr * cfg.min_lr_ratio,
    )

    train_loader = _loader(train_data, cfg, shuffle=True)
    val_loader = _loader(val_data, cfg, shuffle=False)

    best_val_loss = float("inf")
    best_state: Optional[dict] = None
    best_epoch = 0
    no_improve = 0
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch in train_loader:
            inputs, labels = _to_device(batch, device)
            inputs = _augment(inputs, cfg)
            inputs, labels = _mixup(inputs, labels, cfg.mixup_alpha)
            optimizer.zero_grad()
            loss = loss_fn(model(*inputs), _smooth(labels, cfg.label_smoothing))
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        # Linear warmup for the first few epochs, then cosine decay. Warmup
        # matters because the head starts from scratch on a small dataset,
        # where a full-rate first step can wash out the initialisation.
        if epoch <= cfg.warmup_epochs:
            scale = epoch / max(1, cfg.warmup_epochs)
            for group in optimizer.param_groups:
                group["lr"] = max(cfg.lr * cfg.min_lr_ratio, cfg.lr * scale)
        else:
            scheduler.step()

        model.eval()
        val_loss_sum, val_hit_sum, seen = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = _to_device(batch, device)
                logits = model(*inputs)
                count = labels.shape[0]
                val_loss_sum += loss_fn(logits, labels).item() * count
                val_hit_sum += top1_hit_rate(logits, labels) * count
                seen += count
        val_loss = val_loss_sum / max(seen, 1)
        val_hit = val_hit_sum / max(seen, 1)
        history.append({"epoch": epoch, "val_loss": val_loss, "val_top1_hit_rate": val_hit})

        if val_loss < best_val_loss - 1e-6:
            best_val_loss, best_epoch, no_improve = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_run": len(history),
        "history": history,
    }


def evaluate(model: nn.Module, data: Batch, device: Optional[str] = None,
             batch_size: int = 4096) -> float:
    """Top-1 routing accuracy: how often ``argmax(model(x))`` is oracle-correct.

    Evaluated in batches so a large held-out partition, or a cross-attention
    feature carrying a full frame sequence per example, does not have to fit on
    the device all at once.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    inputs, labels = data
    total = labels.shape[0]
    if total == 0:
        return 0.0
    hits = 0.0
    with torch.no_grad():
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            chunk = tuple(t[start:stop].to(device) for t in inputs)
            chunk_labels = labels[start:stop].to(device)
            hits += top1_hit_rate(model(*chunk), chunk_labels) * (stop - start)
    return hits / total
