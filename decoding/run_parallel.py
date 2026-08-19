"""
Parallel launcher: spawns one ``run_qwen``/``run_af3`` subprocess per GPU and
distributes the perturbation registry across them.

Every branch is independent -- its own forward passes, its own output file --
so this is plain process-per-GPU dispatch rather than anything
distributed-training-shaped. Work is handed out dynamically rather than
pre-partitioned, because branch cost varies by more than an order of
magnitude: NO_AUDIO skips the audio encoder entirely, while ``TIMESTRETCH``
and ``PITCH_SHIFT`` do heavy CPU-side resampling before every forward pass. A
static round-robin would leave GPUs idle waiting on the slowest slice.

Branches that already have results on disk are filtered out before dispatch,
so re-running after an interruption costs only the work that was actually
lost. A subprocess is isolated per branch: one CUDA OOM or one corrupt audio
file fails that branch alone, and the launcher reports every failure together
at the end rather than aborting the sweep at the first one.

Usage:
    python -m decoding.run_parallel --model qwen2 --dataset ah_existence --gpus 0 1
    python -m decoding.run_parallel --model af3 --dataset ah_order --gpus 0 1 2 3
    python -m decoding.run_parallel --model qwen2 --dataset ah_existence \
        --prompt aad --perturbation NO_AUDIO ORIGINAL
    python -m decoding.run_parallel --model qwen2 --dataset ah_existence --dry-run
"""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import subprocess
import sys
import time
from collections import deque
from typing import Optional

from helpers.config import DATASETS, PAPER_ALPHA, PROMPTS, RunConfig
from helpers.runtime import output_filename
from perturbations import FAMILY_ORDER, iter_perturbation_configs, spec_label

RUNNERS = {
    "qwen2": "decoding.run_qwen",
    "af3": "decoding.run_af3",
}

POLL_SECONDS = 2.0

_children: list[subprocess.Popen] = []


def _spawn(cmd: list[str], env: dict) -> subprocess.Popen:
    """Start a worker in its own process group so it can be signalled as a unit.

    Without the group, Ctrl-C reaches the launcher but leaves a worker holding
    several gigabytes of GPU memory behind.
    """
    kwargs = {"preexec_fn": os.setsid} if hasattr(os, "setsid") else {}
    process = subprocess.Popen(cmd, env=env, **kwargs)
    _children.append(process)
    return process


def _kill(process: subprocess.Popen, hard: bool) -> None:
    try:
        sig = signal.SIGKILL if hard else signal.SIGTERM
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(process.pid), sig)
        else:
            process.kill() if hard else process.terminate()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _cleanup() -> None:
    """Ask every live worker to stop, then insist."""
    live = [p for p in _children if p.poll() is None]
    if not live:
        return
    for process in live:
        _kill(process, hard=False)
    time.sleep(0.5)
    for process in live:
        if process.poll() is None:
            _kill(process, hard=True)


atexit.register(_cleanup)
signal.signal(signal.SIGINT, lambda *_: (_cleanup(), sys.exit(130)))
signal.signal(signal.SIGTERM, lambda *_: (_cleanup(), sys.exit(143)))


def _select_jobs(args: argparse.Namespace) -> list[tuple[str, Optional[str]]]:
    """Resolve the requested branch set, honouring family and type filters."""
    families = tuple(args.family) if args.family else None
    jobs = list(iter_perturbation_configs(families=families))
    if args.perturbation:
        wanted = {name.upper() for name in args.perturbation}
        jobs = [job for job in jobs if job[0] in wanted]
        unmatched = wanted - {job[0] for job in jobs}
        if unmatched:
            raise SystemExit(f"No branches matched perturbation type(s): {sorted(unmatched)}")
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel perturbation-sweep launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", "-m", required=True, choices=list(RUNNERS))
    parser.add_argument("--dataset", "-d", required=True, choices=list(DATASETS))
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--alpha", "-a", type=float, default=PAPER_ALPHA)
    parser.add_argument("--prompt", default="constrained", choices=sorted(PROMPTS))
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--gpus", "-g", type=int, nargs="+", default=[0],
                        help="CUDA device indices; one worker is kept busy per entry")
    parser.add_argument("--family", nargs="+", default=None, choices=list(FAMILY_ORDER),
                        help="Restrict the sweep to these perturbation families")
    parser.add_argument("--perturbation", "-p", nargs="+", default=None,
                        help="Restrict the sweep to these branch types (all their settings)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the branches that would run, then exit")
    args = parser.parse_args()

    cfg = RunConfig(model=args.model, dataset=args.dataset, alpha=args.alpha,
                    split=args.split, prompt_name=args.prompt)
    if len(args.gpus) != len(set(args.gpus)):
        parser.error("--gpus must list each GPU at most once")

    requested = _select_jobs(args)
    pending = [
        job for job in requested
        if not output_filename(cfg.results_dir, job[0], job[1], cfg.alpha).exists()
    ]
    done = len(requested) - len(pending)

    print(f"{cfg.describe()}")
    print(f"Output directory: {cfg.results_dir}")
    print(f"{len(requested)} branch(es) requested; {done} already complete, {len(pending)} to run.")

    if args.dry_run:
        for pert_type, setting in pending:
            print(f"  would run {spec_label(pert_type, setting)}")
        return
    if not pending:
        print("Nothing to do.")
        return

    jobs = deque(pending)
    total = len(jobs)
    started = 0
    active: dict[int, tuple[subprocess.Popen, str, Optional[str]]] = {}
    failures: list[tuple[int, str, Optional[str], int]] = []

    def launch_next(gpu: int) -> bool:
        nonlocal started
        if not jobs:
            return False
        pert_type, setting = jobs.popleft()
        cmd = [
            sys.executable, "-m", RUNNERS[args.model],
            "--perturbation", pert_type,
            "--dataset", args.dataset,
            "--alpha", str(args.alpha),
            "--prompt", args.prompt,
            "--max-new-tokens", str(args.max_new_tokens),
        ]
        if setting:
            cmd += ["--setting", setting]
        if args.split:
            cmd += ["--split", args.split]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        started += 1
        print(f"[GPU {gpu}] ({started}/{total}) {spec_label(pert_type, setting)}")
        active[gpu] = (_spawn(cmd, env), pert_type, setting)
        return True

    for gpu in args.gpus:
        launch_next(gpu)

    while active:
        for gpu, (process, pert_type, setting) in list(active.items()):
            if process.poll() is None:
                continue
            if process.returncode != 0:
                failures.append((gpu, pert_type, setting, process.returncode))
                print(f"[GPU {gpu}] {spec_label(pert_type, setting)} exited with code {process.returncode}")
            del active[gpu]
            launch_next(gpu)
        time.sleep(POLL_SECONDS)

    if failures:
        details = "\n".join(
            f"  GPU {gpu}  {spec_label(pert_type, setting)}  (exit {code})"
            for gpu, pert_type, setting, code in failures
        )
        raise SystemExit(
            f"{len(failures)} of {total} branch(es) failed:\n{details}\n"
            "Re-run this command to retry only the failures; completed branches are skipped."
        )
    print(f"All {total} perturbation branch(es) complete -> {cfg.results_dir}")


if __name__ == "__main__":
    main()
