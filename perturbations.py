"""
Audio perturbation library for contrastive decoding (paper Section III-B).

Each perturbation is an intentionally destructive transform of a waveform.
During contrastive decoding (see ``decoding/contrastive.py``) a perturbed copy
of the audio stands in for the "negative" branch: the model's next-token
logits on the perturbed audio approximate what it would predict if it could
not fully hear the acoustic evidence the question depends on. Subtracting that
branch from the clean-audio branch amplifies tokens that are actually grounded
in sound.

The design premise is that a negative branch should be *task-oriented*. A
question about temporal order needs a branch that destroys temporal structure
while leaving the events audible; a question about whether a sound is present
at all needs a branch that removes acoustic content outright. Blunt branches
(silence, white noise) cover only one corner of that space, so the library
spans six families:

    Temporal            reordering, stretching, masking along the time axis
    Frequency filters    Butterworth low/high/band pass and stop
    Spectral             operations on the STFT: pitch, blur, reordering, HPSS
    Amplitude & dynamics clipping, quantisation, gating, level normalisation
    Environmental        reverb, echo, and channel simulations
    Additive noise       white and coloured noise at several strengths

Perturbations are organised as classes, not functions, so each family carries
its own named, human-readable settings (e.g. NOISE ``heavy`` vs
``overwhelming``) next to the transform that interprets them, together with
the family tag and the paper-facing label used when regenerating tables. This
is a registry, not a framework: adding a perturbation means adding one class
with an ``apply`` staticmethod and a ``settings`` dict, then listing it in
``ALL_CLASSES``.

Library size: 36 perturbation classes x their settings = 104 configured
specifications, plus the NO_AUDIO baseline (no real audio at all) = 105
perturbations. Counting ORIGINAL (the unperturbed reference branch) alongside
the 36 classes and NO_AUDIO gives 38 branch "types" in total. Run this file
directly to verify both counts and print the per-family breakdown.

Reproducibility contract
------------------------
Most transforms are pure functions of the waveform. Eight classes draw random
numbers (see ``STOCHASTIC_TYPES``): their output depends on NumPy's global
random state, not on the input alone. The decoding runners call
``helpers.runtime.set_random_seed(42)`` at import and then walk the manifest
in a fixed order, so a full branch sweep is reproducible end to end; a single
``apply`` call in isolation is not. Pass an explicit ``rng`` to
``get_perturbation`` when you need a self-contained, order-independent draw.

Implementation notes: frequency filtering uses fifth-order Butterworth filters
(SciPy, zero-phase via ``filtfilt``); pitch shifting, time stretching, STFT
operations, and harmonic/percussive separation use librosa.
"""

from __future__ import annotations

import numpy as np
import librosa
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
from typing import Callable, Dict, Iterator, List, Optional, Tuple

# =============================================================================
# FAMILIES (paper Section III-B)
# =============================================================================

TEMPORAL = "temporal"
FREQUENCY = "frequency"
SPECTRAL = "spectral"
AMPLITUDE = "amplitude"
ENVIRONMENTAL = "environmental"
ADDITIVE_NOISE = "additive_noise"

FAMILY_LABELS = {
    TEMPORAL: "Temporal",
    FREQUENCY: "Frequency filters",
    SPECTRAL: "Spectral",
    AMPLITUDE: "Amplitude & dynamics",
    ENVIRONMENTAL: "Environmental",
    ADDITIVE_NOISE: "Additive noise",
}

FAMILY_ORDER: Tuple[str, ...] = (
    TEMPORAL, FREQUENCY, SPECTRAL, AMPLITUDE, ENVIRONMENTAL, ADDITIVE_NOISE,
)

DEFAULT_SR = 16000


def _rng(rng: Optional[np.random.Generator]):
    """Return the caller's Generator, or NumPy's global legacy state.

    The legacy fallback is what the reported runs used: the runners seed
    ``np.random`` once per process, so the draw sequence is fixed by the
    evaluation order. ``np.random`` and ``Generator`` share the method names
    used below (``standard_normal``, ``random``, ``shuffle``), except for
    integer sampling, which is handled by ``_randint``.
    """
    return np.random if rng is None else rng


def _randint(source, low: int, high: int) -> int:
    """Draw one integer in [low, high) from either RNG flavour."""
    if source is np.random:
        return int(source.randint(low, high))
    return int(source.integers(low, high))


# =============================================================================
# PERTURBATION CLASSES
#
# Every class declares:
#   name             registry key, also the label written into result files
#   label            human-readable name used in regenerated paper tables
#   family           one of the six constants above
#   stochastic       True when apply() consumes random numbers
#   settings         {setting_name: kwargs passed to apply()}
#   setting_labels   optional paper wording for a setting, keyed by name
# =============================================================================

class NoisePerturbation:
    name = "NOISE"
    label = "Noise"
    family = ADDITIVE_NOISE
    stochastic = True
    description = "Gaussian white noise added at sigma relative to waveform amplitude."
    settings = {
        "heavy":        {"sigma": 0.3},
        "very_heavy":   {"sigma": 0.5},
        "extreme":      {"sigma": 0.6},
        "overwhelming": {"sigma": 1.0},
    }

    @staticmethod
    def apply(audio: np.ndarray, sigma: float = 0.3, sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        noise = _rng(rng).standard_normal(audio.shape).astype(np.float32) * sigma
        return audio + noise


class ColoredNoisePerturbation:
    name = "COLORED_NOISE"
    label = "Colored noise"
    family = ADDITIVE_NOISE
    stochastic = True
    description = "Pink (1/f) or brown (integrated) noise added at sigma relative to waveform amplitude."
    settings = {
        "pink_heavy":  {"color": "pink",  "sigma": 0.4},
        "brown_heavy": {"color": "brown", "sigma": 0.5},
    }

    @staticmethod
    def apply(audio: np.ndarray, color: str = "pink", sigma: float = 0.4,
              sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        n = len(audio)
        white = _rng(rng).standard_normal(n).astype(np.float32)
        if color == "pink":
            # 1/f shaping in the frequency domain; bin 0 is held at unit
            # divisor so the DC term is not amplified without bound.
            fft = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(n)
            freqs[0] = 1
            fft = fft / np.sqrt(freqs)
            noise = np.fft.irfft(fft, n).astype(np.float32)
        elif color == "brown":
            noise = np.cumsum(white).astype(np.float32)
            noise = noise - np.mean(noise)
        else:
            raise ValueError(f"color must be 'pink' or 'brown', got {color!r}")
        noise = noise / (np.std(noise) + 1e-8) * sigma
        return audio + noise


class TimestretchPerturbation:
    name = "TIMESTRETCH"
    label = "Timestretch"
    family = TEMPORAL
    stochastic = False
    description = "Phase-vocoder time stretch; pitch preserved, duration changed."
    settings = {
        "very_slow": {"rate": 0.4},
        "very_fast": {"rate": 2.5},
    }
    setting_labels = {"very_slow": "0.4x", "very_fast": "2.5x"}

    @staticmethod
    def apply(audio: np.ndarray, rate: float = 0.5, sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        return librosa.effects.time_stretch(audio, rate=rate)


class SegmentShufflePerturbation:
    name = "SEGMENT_SHUFFLE"
    label = "Segment shuffle"
    family = TEMPORAL
    stochastic = True
    description = "Waveform cut into K equal blocks and randomly reordered; content preserved, order destroyed."
    settings = {
        "coarse":  {"n_segments": 10},
        "fine":    {"n_segments": 50},
        "extreme": {"n_segments": 200},
    }
    setting_labels = {"coarse": "10 seg", "fine": "50 seg", "extreme": "200 seg"}

    @staticmethod
    def apply(audio: np.ndarray, n_segments: int = 50, sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        segment_len = len(audio) // n_segments
        if segment_len < 1:
            return audio
        segments = [audio[i * segment_len:(i + 1) * segment_len] for i in range(n_segments)]
        remainder = audio[n_segments * segment_len:]
        _rng(rng).shuffle(segments)
        result = np.concatenate(segments)
        if len(remainder) > 0:
            result = np.concatenate([result, remainder])
        return result


class SegmentReversePerturbation:
    name = "SEGMENT_REVERSE"
    label = "Segment reverse"
    family = TEMPORAL
    stochastic = False
    description = "Each of K equal blocks reversed in place; block order kept, within-block direction flipped."
    settings = {
        "coarse": {"n_segments": 10},
        "fine":   {"n_segments": 50},
    }
    setting_labels = {"coarse": "10 seg", "fine": "50 seg"}

    @staticmethod
    def apply(audio: np.ndarray, n_segments: int = 10, sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        segment_len = len(audio) // n_segments
        if segment_len < 1:
            return audio
        result = audio.copy()
        for i in range(n_segments):
            start = i * segment_len
            end = (i + 1) * segment_len
            result[start:end] = result[start:end][::-1]
        return result


class ReversePerturbation:
    name = "REVERSE"
    label = "Reverse"
    family = TEMPORAL
    stochastic = False
    description = (
        "Flips the audio array end to end so the clip plays backwards. The paper's "
        "strongest AH Order branch: +6.7 points for AF3 (74.7% -> 81.4%)."
    )
    settings = {"full": {}}
    setting_labels = {"full": ""}

    @staticmethod
    def apply(audio: np.ndarray, sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        return audio[::-1].copy()


class DropoutPerturbation:
    name = "DROPOUT"
    label = "Dropout"
    family = TEMPORAL
    stochastic = True
    description = "Individual samples zero-masked independently with probability p."
    settings = {
        "heavy":   {"p": 0.4},
        "extreme": {"p": 0.7},
    }
    setting_labels = {"heavy": "p=0.4", "extreme": "p=0.7"}

    @staticmethod
    def apply(audio: np.ndarray, p: float = 0.5, sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        mask = _rng(rng).random(len(audio)) > p
        result = audio.copy()
        result[~mask] = 0
        return result


class TimeMaskPerturbation:
    name = "TIME_MASK"
    label = "Time mask"
    family = TEMPORAL
    stochastic = True
    description = "Contiguous silence intervals, each up to max_width of the clip duration."
    settings = {
        "light":   {"n_masks": 3,  "max_width": 0.08},
        "heavy":   {"n_masks": 5,  "max_width": 0.15},
        "extreme": {"n_masks": 10, "max_width": 0.10},
    }

    @staticmethod
    def apply(audio: np.ndarray, n_masks: int = 5, max_width: float = 0.15,
              sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        source = _rng(rng)
        result = audio.copy()
        audio_len = len(audio)
        for _ in range(n_masks):
            width = int(source.uniform(0.05, max_width) * audio_len)
            start = _randint(source, 0, max(1, audio_len - width))
            result[start:start + width] = 0
        return result


class RepeatSegmentPerturbation:
    name = "REPEAT_SEGMENT"
    label = "Repeat segment"
    family = TEMPORAL
    stochastic = False
    description = "A 20% slice (from the start or the middle) looped to fill the original duration."
    settings = {
        "repeat_start":  {"segment": "start",  "repeats": 5},
        "repeat_middle": {"segment": "middle", "repeats": 5},
    }
    setting_labels = {"repeat_start": "start", "repeat_middle": "middle"}

    @staticmethod
    def apply(audio: np.ndarray, segment: str = "start", repeats: int = 5,
              sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        segment_len = len(audio) // 5
        if segment == "start":
            seg = audio[:segment_len]
        elif segment == "middle":
            mid = len(audio) // 2
            seg = audio[mid - segment_len // 2:mid + segment_len // 2]
        else:
            seg = audio[-segment_len:]
        return np.tile(seg, repeats)[:len(audio)]


class LowPassPerturbation:
    name = "LOW_PASS"
    label = "Low pass"
    family = FREQUENCY
    stochastic = False
    description = "Fifth-order Butterworth low-pass; everything above the cutoff is rolled off."
    settings = {
        "extreme":    {"cutoff_hz": 250},
        "aggressive": {"cutoff_hz": 500},
        "moderate":   {"cutoff_hz": 1000},
    }
    setting_labels = {"extreme": "250 Hz", "aggressive": "500 Hz", "moderate": "1 kHz"}

    @staticmethod
    def apply(audio: np.ndarray, cutoff_hz: float = 500, sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        nyq = sr / 2
        normalized_cutoff = min(cutoff_hz / nyq, 0.99)
        b, a = signal.butter(5, normalized_cutoff, btype="low")
        return signal.filtfilt(b, a, audio).astype(np.float32)


class HighPassPerturbation:
    name = "HIGH_PASS"
    label = "High pass"
    family = FREQUENCY
    stochastic = False
    description = "Fifth-order Butterworth high-pass; everything below the cutoff is rolled off."
    settings = {
        "moderate":      {"cutoff_hz": 1000},
        "aggressive":    {"cutoff_hz": 2000},
        "extreme":       {"cutoff_hz": 4000},
        "very_extreme":  {"cutoff_hz": 5000},
        "ultra_extreme": {"cutoff_hz": 6000},
    }
    setting_labels = {
        "moderate": "1 kHz", "aggressive": "2 kHz", "extreme": "4 kHz",
        "very_extreme": "5 kHz", "ultra_extreme": "6 kHz",
    }

    @staticmethod
    def apply(audio: np.ndarray, cutoff_hz: float = 2000, sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        nyq = sr / 2
        normalized_cutoff = min(cutoff_hz / nyq, 0.99)
        b, a = signal.butter(5, normalized_cutoff, btype="high")
        return signal.filtfilt(b, a, audio).astype(np.float32)


class BandpassPerturbation:
    name = "BANDPASS"
    label = "Bandpass"
    family = FREQUENCY
    stochastic = False
    description = "Fifth-order Butterworth bandpass; isolates one band and discards the rest."
    settings = {
        "bass_only":       {"low_hz": 50,   "high_hz": 300},
        "bass_wide":       {"low_hz": 50,   "high_hz": 500},
        "low_mid":         {"low_hz": 200,  "high_hz": 800},
        "mid_narrow":      {"low_hz": 700,  "high_hz": 1400},
        "mid_only":        {"low_hz": 500,  "high_hz": 2000},
        "high_mid":        {"low_hz": 1500, "high_hz": 4000},
        "high_mid_narrow": {"low_hz": 2000, "high_hz": 3500},
        "treble_only":     {"low_hz": 3000, "high_hz": 8000},
        "treble_extreme":  {"low_hz": 4000, "high_hz": 8000},
        "treble_ultra":    {"low_hz": 5000, "high_hz": 8000},
    }

    @staticmethod
    def apply(audio: np.ndarray, low_hz: float = 500, high_hz: float = 2000,
              sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        nyq = sr / 2
        low = min(low_hz / nyq, 0.99)
        high = min(high_hz / nyq, 0.99)
        if low >= high:
            low = high - 0.01
        b, a = signal.butter(5, [low, high], btype="band")
        return signal.filtfilt(b, a, audio).astype(np.float32)


class BandstopPerturbation:
    name = "BANDSTOP"
    label = "Bandstop"
    family = FREQUENCY
    stochastic = False
    description = "Fifth-order Butterworth band-reject; notches one band and keeps the rest."
    settings = {
        "remove_low":  {"low_hz": 50,   "high_hz": 500},
        "remove_mids": {"low_hz": 500,  "high_hz": 2000},
        "remove_high": {"low_hz": 3000, "high_hz": 8000},
    }

    @staticmethod
    def apply(audio: np.ndarray, low_hz: float = 500, high_hz: float = 2000,
              sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        nyq = sr / 2
        low = min(low_hz / nyq, 0.99)
        high = min(high_hz / nyq, 0.99)
        if low >= high:
            low = high - 0.01
        b, a = signal.butter(5, [low, high], btype="bandstop")
        return signal.filtfilt(b, a, audio).astype(np.float32)


class FreqMaskPerturbation:
    name = "FREQ_MASK"
    label = "Frequency mask"
    family = FREQUENCY
    stochastic = True
    description = "Multiple random STFT frequency bands zeroed, then reconstructed via ISTFT."
    settings = {
        "heavy":      {"n_masks": 8,  "max_width_hz": 500},
        "very_heavy": {"n_masks": 12, "max_width_hz": 600},
        "extreme":    {"n_masks": 15, "max_width_hz": 400},
    }

    @staticmethod
    def apply(audio: np.ndarray, n_masks: int = 8, max_width_hz: float = 500,
              sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        source = _rng(rng)
        stft = librosa.stft(audio)
        n_bins = stft.shape[0]
        hz_per_bin = (sr / 2) / n_bins
        for _ in range(n_masks):
            width_bins = int(source.uniform(100, max_width_hz) / hz_per_bin)
            start = _randint(source, 0, max(1, n_bins - width_bins))
            stft[start:start + width_bins, :] = 0
        return librosa.istft(stft, length=len(audio)).astype(np.float32)


class PitchShiftPerturbation:
    name = "PITCH_SHIFT"
    label = "Pitch shift"
    family = SPECTRAL
    stochastic = False
    description = (
        "Resampling pitch shift with duration preserved. The paper's strongest AF3 "
        "AH Existence branch at +24 semitones (69.5% -> 73.9%)."
    )
    settings = {
        "extreme_down":  {"semitones": -24},
        "down_octave":   {"semitones": -12},
        "down_moderate": {"semitones": -6},
        "down_mild":     {"semitones": -4},
        "up_mild":       {"semitones": 4},
        "up_moderate":   {"semitones": 6},
        "up_octave":     {"semitones": 12},
        "extreme_up":    {"semitones": 24},
    }
    setting_labels = {
        "extreme_down": "down two octaves", "down_octave": "down octave",
        "up_octave": "up octave", "extreme_up": "up two octaves",
    }

    @staticmethod
    def apply(audio: np.ndarray, semitones: int = 12, sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)


class SpecNoisePerturbation:
    name = "SPEC_NOISE"
    label = "Spectral noise"
    family = SPECTRAL
    stochastic = True
    description = (
        "Gaussian noise injected into the STFT magnitude at sigma relative to its RMS; "
        "the original phase is retained."
    )
    settings = {
        "light":        {"sigma": 0.1},
        "heavy":        {"sigma": 0.3},
        "extreme":      {"sigma": 0.6},
        "overwhelming": {"sigma": 1.0},
    }

    @staticmethod
    def apply(audio: np.ndarray, sigma: float = 0.3, sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        n_fft, hop_length = 1024, 256
        stft = librosa.stft(audio.astype(np.float32), n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.exp(1j * np.angle(stft))
        rms = float(np.sqrt(np.mean(magnitude ** 2))) + 1e-8
        noise = _rng(rng).standard_normal(magnitude.shape).astype(np.float32) * sigma * rms
        magnitude_noisy = np.maximum(magnitude + noise, 0.0)
        return librosa.istft(magnitude_noisy * phase, hop_length=hop_length,
                             length=len(audio)).astype(np.float32)


class SpectralBlurPerturbation:
    name = "SPECTRAL_BLUR"
    label = "Spectral blur"
    family = SPECTRAL
    stochastic = False
    description = (
        "Sequential 1-D Gaussian smoothing of the STFT magnitude along frequency then "
        "time. Sigma is measured in STFT bins and shared across both axes."
    )
    settings = {
        "light":   {"sigma_time": 5,  "sigma_freq": 5},
        "heavy":   {"sigma_time": 15, "sigma_freq": 15},
        "extreme": {"sigma_time": 25, "sigma_freq": 25},
    }
    setting_labels = {"light": "sigma=5", "heavy": "sigma=15", "extreme": "sigma=25"}

    @staticmethod
    def apply(audio: np.ndarray, sigma_time: float = 10, sigma_freq: float = 10,
              sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        stft = librosa.stft(audio)
        mag = np.abs(stft)
        phase = np.angle(stft)
        mag_blurred = gaussian_filter1d(mag, sigma=sigma_freq, axis=0)
        mag_blurred = gaussian_filter1d(mag_blurred, sigma=sigma_time, axis=1)
        return librosa.istft(mag_blurred * np.exp(1j * phase), length=len(audio)).astype(np.float32)


class SpecReversePerturbation:
    name = "SPEC_REVERSE"
    label = "Spectral reverse"
    family = SPECTRAL
    stochastic = False
    description = "STFT frame columns reversed end to end, then reconstructed via ISTFT."
    settings = {"full": {}}
    setting_labels = {"full": ""}

    @staticmethod
    def apply(audio: np.ndarray, sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        stft = librosa.stft(audio.astype(np.float32), n_fft=1024, hop_length=256)
        return librosa.istft(stft[:, ::-1], hop_length=256, length=len(audio)).astype(np.float32)


class SpecSegmentReversePerturbation:
    name = "SPEC_SEGMENT_REVERSE"
    label = "Spec. seg. reverse"
    family = SPECTRAL
    stochastic = False
    description = "Blocks of STFT frame columns reversed in place, then reconstructed via ISTFT."
    settings = {
        "coarse": {"n_segments": 10},
        "fine":   {"n_segments": 50},
    }
    setting_labels = {"coarse": "10 seg", "fine": "50 seg"}

    @staticmethod
    def apply(audio: np.ndarray, n_segments: int = 10, sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        stft = librosa.stft(audio.astype(np.float32), n_fft=1024, hop_length=256)
        n_frames = stft.shape[1]
        seg_len = n_frames // n_segments
        if seg_len < 1:
            return audio
        result = stft.copy()
        for i in range(n_segments):
            start = i * seg_len
            end = (i + 1) * seg_len
            result[:, start:end] = stft[:, start:end][:, ::-1]
        return librosa.istft(result, hop_length=256, length=len(audio)).astype(np.float32)


class SpecSegmentShufflePerturbation:
    name = "SPEC_SEGMENT_SHUFFLE"
    label = "Spec. seg. shuffle"
    family = SPECTRAL
    stochastic = True
    description = "Blocks of STFT frame columns randomly reordered, then reconstructed via ISTFT."
    settings = {
        "coarse":  {"n_segments": 10},
        "fine":    {"n_segments": 50},
        "extreme": {"n_segments": 200},
    }
    setting_labels = {"coarse": "10 seg", "fine": "50 seg", "extreme": "200 seg"}

    @staticmethod
    def apply(audio: np.ndarray, n_segments: int = 50, sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        stft = librosa.stft(audio.astype(np.float32), n_fft=1024, hop_length=256)
        n_frames = stft.shape[1]
        seg_len = n_frames // n_segments
        if seg_len < 1:
            return audio
        segments = [stft[:, i * seg_len:(i + 1) * seg_len] for i in range(n_segments)]
        remainder = stft[:, n_segments * seg_len:]
        _rng(rng).shuffle(segments)
        result = np.concatenate(segments, axis=1)
        if remainder.size:
            result = np.concatenate([result, remainder], axis=1)
        return librosa.istft(result, hop_length=256, length=len(audio)).astype(np.float32)


class HarmonicRemovePerturbation:
    name = "HARMONIC_REMOVE"
    label = "Harmonic remove"
    family = SPECTRAL
    stochastic = False
    description = "Librosa median-filter HPSS; keeps only the percussive (transient) component."
    settings = {"full": {"margin": 3.0}}
    setting_labels = {"full": ""}

    @staticmethod
    def apply(audio: np.ndarray, margin: float = 3.0, sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        _, percussive = librosa.effects.hpss(audio, margin=margin)
        return percussive


class PercussiveRemovePerturbation:
    name = "PERCUSSIVE_REMOVE"
    label = "Percussive remove"
    family = SPECTRAL
    stochastic = False
    description = "Librosa median-filter HPSS; keeps only the harmonic (tonal) component."
    settings = {"full": {"margin": 3.0}}
    setting_labels = {"full": ""}

    @staticmethod
    def apply(audio: np.ndarray, margin: float = 3.0, sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        harmonic, _ = librosa.effects.hpss(audio, margin=margin)
        return harmonic


class ClipPerturbation:
    name = "CLIP"
    label = "Clip"
    family = AMPLITUDE
    stochastic = False
    description = "Hard amplitude truncation to +/- threshold, flattening every peak above it."
    settings = {
        "extreme": {"threshold": 0.1},
        "hard":    {"threshold": 0.2},
    }
    setting_labels = {"extreme": "thr=0.1", "hard": "thr=0.2"}

    @staticmethod
    def apply(audio: np.ndarray, threshold: float = 0.2, sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        return np.clip(audio, -threshold, threshold)


class QuantizePerturbation:
    name = "QUANTIZE"
    label = "Quantize"
    family = AMPLITUDE
    stochastic = False
    description = "Bit-depth reduction to 2-4 bits over the clip's own amplitude range."
    settings = {
        "4bit": {"bits": 4},
        "3bit": {"bits": 3},
        "2bit": {"bits": 2},
    }
    setting_labels = {"4bit": "4-bit", "3bit": "3-bit", "2bit": "2-bit"}

    @staticmethod
    def apply(audio: np.ndarray, bits: int = 3, sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        levels = 2 ** bits
        span = audio.max() - audio.min()
        audio_norm = (audio - audio.min()) / (span + 1e-8)
        quantized = np.round(audio_norm * (levels - 1)) / (levels - 1)
        return (quantized * span + audio.min()).astype(np.float32)


class CompressPerturbation:
    name = "COMPRESS"
    label = "Compress"
    family = AMPLITUDE
    stochastic = False
    description = "Dynamic-range compression: amplitude above the threshold is scaled down by ratio."
    settings = {
        "heavy":   {"threshold": 0.2, "ratio": 10},
        "extreme": {"threshold": 0.1, "ratio": 20},
    }
    setting_labels = {"heavy": "10:1", "extreme": "20:1"}

    @staticmethod
    def apply(audio: np.ndarray, threshold: float = 0.2, ratio: float = 10,
              sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        result = audio.copy()
        above_thresh = np.abs(audio) > threshold
        excess = np.abs(audio[above_thresh]) - threshold
        result[above_thresh] = np.sign(audio[above_thresh]) * (threshold + excess / ratio)
        return result


class GatePerturbation:
    name = "GATE"
    label = "Gate"
    family = AMPLITUDE
    stochastic = False
    description = "Noise gate: samples quieter than the threshold are zeroed, keeping only peaks."
    settings = {
        "aggressive":    {"threshold": 0.30},
        "extreme":       {"threshold": 0.50},
        "very_extreme":  {"threshold": 0.65},
        "ultra_extreme": {"threshold": 0.75},
    }
    setting_labels = {
        "aggressive": "thr=0.30", "extreme": "thr=0.50",
        "very_extreme": "thr=0.65", "ultra_extreme": "thr=0.75",
    }

    @staticmethod
    def apply(audio: np.ndarray, threshold: float = 0.3, sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        result = audio.copy()
        result[np.abs(audio) < threshold] = 0
        return result


class GateInvertedPerturbation:
    name = "GATE_INVERTED"
    label = "Gate inv."
    family = AMPLITUDE
    stochastic = False
    description = "Inverted gate: samples louder than the threshold are zeroed, keeping only the quiet floor."
    settings = {
        "moderate":     {"threshold": 0.50},
        "aggressive":   {"threshold": 0.30},
        "extreme":      {"threshold": 0.20},
        "very_extreme": {"threshold": 0.10},
    }
    setting_labels = {
        "moderate": "thr=0.50", "aggressive": "thr=0.30",
        "extreme": "thr=0.20", "very_extreme": "thr=0.10",
    }

    @staticmethod
    def apply(audio: np.ndarray, threshold: float = 0.3, sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        result = audio.copy()
        result[np.abs(audio) >= threshold] = 0
        return result


class GateSoftPerturbation:
    name = "GATE_SOFT"
    label = "Gate soft"
    family = AMPLITUDE
    stochastic = False
    description = "Gate with attenuation instead of hard zeroing; quiet samples are ducked by attenuation_db."
    settings = {
        "soft_aggressive":   {"threshold": 0.3, "attenuation_db": -18},
        "soft_extreme":      {"threshold": 0.3, "attenuation_db": -30},
        "soft_very_extreme": {"threshold": 0.5, "attenuation_db": -24},
        "soft_ultra":        {"threshold": 0.5, "attenuation_db": -45},
    }

    @staticmethod
    def apply(audio: np.ndarray, threshold: float = 0.3, attenuation_db: float = -24,
              sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        result = audio.copy()
        gain = 10 ** (attenuation_db / 20)
        result[np.abs(audio) < threshold] *= gain
        return result


class GateInvertedSoftPerturbation:
    name = "GATE_INVERTED_SOFT"
    label = "Gate inv. soft"
    family = AMPLITUDE
    stochastic = False
    description = "Inverted gate with attenuation instead of hard zeroing; loud samples are ducked."
    settings = {
        "soft_moderate":   {"threshold": 0.5, "attenuation_db": -12},
        "soft_aggressive": {"threshold": 0.3, "attenuation_db": -18},
        "soft_extreme":    {"threshold": 0.2, "attenuation_db": -30},
    }

    @staticmethod
    def apply(audio: np.ndarray, threshold: float = 0.3, attenuation_db: float = -18,
              sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        result = audio.copy()
        gain = 10 ** (attenuation_db / 20)
        result[np.abs(audio) >= threshold] *= gain
        return result


class NormalizeChunksPerturbation:
    name = "NORMALIZE_CHUNKS"
    label = "Normalize chunks"
    family = AMPLITUDE
    stochastic = False
    description = "Peak-normalises K independent temporal chunks, flattening relative loudness over time."
    settings = {
        "coarse": {"n_chunks": 10},
        "fine":   {"n_chunks": 50},
    }
    setting_labels = {"coarse": "10 chunks", "fine": "50 chunks"}

    @staticmethod
    def apply(audio: np.ndarray, n_chunks: int = 20, sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        chunk_len = len(audio) // n_chunks
        if chunk_len < 1:
            return audio
        result = audio.copy()
        for i in range(n_chunks):
            start = i * chunk_len
            end = (i + 1) * chunk_len
            chunk = result[start:end]
            result[start:end] = chunk / (np.max(np.abs(chunk)) + 1e-8)
        return result


class ResampleLowPerturbation:
    name = "RESAMPLE_LOW"
    label = "Resample low"
    family = AMPLITUDE
    stochastic = False
    description = "Downsample to target_sr then upsample back, discarding everything above the new Nyquist."
    settings = {
        "8khz": {"target_sr": 8000},
        "4khz": {"target_sr": 4000},
        "2khz": {"target_sr": 2000},
    }
    setting_labels = {"8khz": "8 kHz", "4khz": "4 kHz", "2khz": "2 kHz"}

    @staticmethod
    def apply(audio: np.ndarray, target_sr: int = 4000, sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        downsampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        upsampled = librosa.resample(downsampled, orig_sr=target_sr, target_sr=sr)
        if len(upsampled) > len(audio):
            upsampled = upsampled[:len(audio)]
        elif len(upsampled) < len(audio):
            upsampled = np.pad(upsampled, (0, len(audio) - len(upsampled)))
        return upsampled


class BitCrushPerturbation:
    name = "BIT_CRUSH"
    label = "Bit crush"
    family = AMPLITUDE
    stochastic = False
    description = "Resample-low followed by bit-depth reduction: both time and amplitude resolution drop."
    settings = {
        "retro":   {"bits": 4, "target_sr": 8000},
        "extreme": {"bits": 2, "target_sr": 4000},
    }
    setting_labels = {"retro": "4-bit, 8 kHz", "extreme": "2-bit, 4 kHz"}

    @staticmethod
    def apply(audio: np.ndarray, bits: int = 3, target_sr: int = 4000,
              sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        resampled = ResampleLowPerturbation.apply(audio, target_sr=target_sr, sr=sr)
        return QuantizePerturbation.apply(resampled, bits=bits, sr=sr)


class ReverbPerturbation:
    name = "REVERB"
    label = "Reverb"
    family = ENVIRONMENTAL
    stochastic = False
    description = "Five-tap comb delay with exponential decay, renormalised to the original peak."
    settings = {
        "large_hall": {"decay": 0.80, "delay": 0.05},
        "extreme":    {"decay": 0.95, "delay": 0.10},
    }
    setting_labels = {"large_hall": "dec=0.80, 50 ms", "extreme": "dec=0.95, 100 ms"}

    @staticmethod
    def apply(audio: np.ndarray, decay: float = 0.8, delay: float = 0.05,
              sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        delay_samples = int(delay * sr)
        result = audio.copy()
        for i in range(5):
            d = delay_samples * (i + 1)
            gain = decay ** (i + 1)
            if 0 < d < len(audio):
                result[d:] += audio[:-d] * gain
        result = result / (np.max(np.abs(result)) + 1e-8) * np.max(np.abs(audio))
        return result.astype(np.float32)


class EchoPerturbation:
    name = "ECHO"
    label = "Echo"
    family = ENVIRONMENTAL
    stochastic = False
    description = "Discrete repeated delays at delay_ms with exponential decay, truncated to the original length."
    settings = {
        "short": {"delay_ms": 100, "decay": 0.6, "repeats": 5},
        "long":  {"delay_ms": 300, "decay": 0.7, "repeats": 8},
    }
    setting_labels = {"short": "100 ms", "long": "300 ms"}

    @staticmethod
    def apply(audio: np.ndarray, delay_ms: float = 200, decay: float = 0.6,
              repeats: int = 5, sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        delay_samples = int(delay_ms / 1000 * sr)
        result = np.zeros(len(audio) + delay_samples * repeats, dtype=np.float32)
        result[:len(audio)] = audio
        for i in range(repeats):
            offset = delay_samples * (i + 1)
            gain = decay ** (i + 1)
            if offset < len(result):
                end_idx = min(offset + len(audio), len(result))
                result[offset:end_idx] += audio[:end_idx - offset] * gain
        return result[:len(audio)]


class PhoneFilterPerturbation:
    name = "PHONE_FILTER"
    label = "Phone filter"
    family = ENVIRONMENTAL
    stochastic = False
    description = "Narrowband telephony simulation: bandpass restricted to the 300-3400 Hz voice band."
    settings = {
        "standard": {"low_hz": 300, "high_hz": 3400},
        "narrow":   {"low_hz": 500, "high_hz": 2500},
    }
    setting_labels = {"standard": "300 Hz-3.4 kHz", "narrow": "500 Hz-2.5 kHz"}

    @staticmethod
    def apply(audio: np.ndarray, low_hz: float = 300, high_hz: float = 3400,
              sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        return BandpassPerturbation.apply(audio, low_hz=low_hz, high_hz=high_hz, sr=sr)


class UnderwaterPerturbation:
    name = "UNDERWATER"
    label = "Underwater"
    family = ENVIRONMENTAL
    stochastic = False
    description = "Low-pass at 400 Hz followed by reverb, simulating a heavily muffled channel."
    settings = {"deep": {"cutoff_hz": 400, "reverb_decay": 0.7}}
    setting_labels = {"deep": "400 Hz"}

    @staticmethod
    def apply(audio: np.ndarray, cutoff_hz: float = 400, reverb_decay: float = 0.7,
              sr: int = DEFAULT_SR, rng=None) -> np.ndarray:
        filtered = LowPassPerturbation.apply(audio, cutoff_hz=cutoff_hz, sr=sr)
        return ReverbPerturbation.apply(filtered, decay=reverb_decay, delay=0.03, sr=sr)


# =============================================================================
# REGISTRY & PUBLIC API
# =============================================================================

ALL_CLASSES: List[type] = [
    # Temporal
    TimestretchPerturbation,
    SegmentShufflePerturbation,
    SegmentReversePerturbation,
    ReversePerturbation,
    DropoutPerturbation,
    TimeMaskPerturbation,
    RepeatSegmentPerturbation,
    # Frequency filters
    LowPassPerturbation,
    HighPassPerturbation,
    BandpassPerturbation,
    BandstopPerturbation,
    FreqMaskPerturbation,
    # Spectral
    PitchShiftPerturbation,
    SpecNoisePerturbation,
    SpectralBlurPerturbation,
    SpecReversePerturbation,
    SpecSegmentReversePerturbation,
    SpecSegmentShufflePerturbation,
    HarmonicRemovePerturbation,
    PercussiveRemovePerturbation,
    # Amplitude & dynamics
    ClipPerturbation,
    QuantizePerturbation,
    CompressPerturbation,
    GatePerturbation,
    GateInvertedPerturbation,
    GateSoftPerturbation,
    GateInvertedSoftPerturbation,
    NormalizeChunksPerturbation,
    ResampleLowPerturbation,
    BitCrushPerturbation,
    # Environmental
    ReverbPerturbation,
    EchoPerturbation,
    PhoneFilterPerturbation,
    UnderwaterPerturbation,
    # Additive noise
    NoisePerturbation,
    ColoredNoisePerturbation,
]

PERTURBATION_SPECS: Dict[type, Dict] = {cls: cls.settings for cls in ALL_CLASSES}
PERTURBATION_BY_NAME: Dict[str, type] = {cls.name: cls for cls in ALL_CLASSES}

# NO_AUDIO drops the audio entirely (the text-only negative branch AAD found
# strongest). ORIGINAL is the unperturbed reference branch, not itself a
# "perturbation" -- see the 104 + 1 = 105 count in the module docstring.
# ORIGINAL is nonetheless a legal *selector candidate*: the paper's AF3 AH
# Existence N=4 pool contains it, meaning "apply no correction here".
BASELINE_PERTURBATIONS: Tuple[str, ...] = ("ORIGINAL", "NO_AUDIO")

BASELINE_LABELS = {"ORIGINAL": "Original", "NO_AUDIO": "No-Audio"}

STOCHASTIC_TYPES: Tuple[str, ...] = tuple(
    cls.name for cls in ALL_CLASSES if cls.stochastic
)

# Every branch name that can legally appear in a result file or candidate pool.
ALL_TYPES: Tuple[str, ...] = BASELINE_PERTURBATIONS + tuple(PERTURBATION_BY_NAME)


def iter_perturbation_configs(
    include_baselines: bool = True,
    families: Optional[Tuple[str, ...]] = None,
) -> Iterator[Tuple[str, Optional[str]]]:
    """Yield ``(type, setting)`` pairs for every branch in the library.

    With ``families`` set, only classes in those families are yielded; the
    baselines are still included unless ``include_baselines`` is False, since
    ORIGINAL and NO_AUDIO belong to no acoustic family.
    """
    if families is not None:
        unknown = sorted(set(families) - set(FAMILY_ORDER))
        if unknown:
            raise ValueError(f"Unknown perturbation family/families {unknown}; expected {list(FAMILY_ORDER)}")
    if include_baselines:
        for name in BASELINE_PERTURBATIONS:
            yield name, None
    for cls in ALL_CLASSES:
        if families is not None and cls.family not in families:
            continue
        for setting in cls.settings:
            yield cls.name, setting


def get_perturbation_class(name: str) -> type:
    """Look up a perturbation class by registry name (case-insensitive)."""
    normalized = name.upper()
    if normalized in BASELINE_PERTURBATIONS:
        raise KeyError(
            f"{normalized!r} is a baseline branch, not a waveform transform; "
            "the decoding runners handle it before reaching the registry."
        )
    try:
        return PERTURBATION_BY_NAME[normalized]
    except KeyError:
        valid = ", ".join(ALL_TYPES)
        raise KeyError(f"Unknown perturbation type {name!r}; expected one of: {valid}") from None


def get_perturbation(cls: type, setting: Optional[str] = None,
                     sr: int = DEFAULT_SR, rng=None) -> Callable:
    """Return a ready-to-call ``fn(audio) -> perturbed_audio`` closure.

    ``rng`` is threaded through to the stochastic transforms; leaving it None
    uses NumPy's global state, which is what the reported runs did.
    """
    if setting is not None and setting not in cls.settings:
        valid = ", ".join(cls.settings) or "(none)"
        raise KeyError(f"Unknown setting {setting!r} for {cls.name}; expected one of: {valid}")
    params = cls.settings[setting].copy() if setting is not None else {}
    return lambda audio: cls.apply(audio, sr=sr, rng=rng, **params)


def spec_label(perturbation_type: str, perturbation_setting: Optional[str]) -> str:
    """Canonical registry label: ``TYPE`` or ``TYPE:setting``.

    This is the identifier used in result files, oracle branch lists, split
    metadata, and selector candidate pools.
    """
    return perturbation_type if perturbation_setting is None else f"{perturbation_type}:{perturbation_setting}"


def paper_label(perturbation_type: str, perturbation_setting: Optional[str] = None) -> str:
    """Human-readable name in the paper's table style, e.g. ``Pitch shift (up two octaves)``.

    Falls back to rendering the setting's raw parameters when no curated
    wording exists, so a newly added setting still prints something meaningful.
    """
    normalized = perturbation_type.upper()
    if normalized in BASELINE_LABELS:
        return BASELINE_LABELS[normalized]
    cls = PERTURBATION_BY_NAME.get(normalized)
    if cls is None:
        return spec_label(perturbation_type, perturbation_setting)
    if perturbation_setting is None:
        return cls.label

    curated = getattr(cls, "setting_labels", {}).get(perturbation_setting)
    if curated is not None:
        return f"{cls.label} ({curated})" if curated else cls.label

    params = cls.settings.get(perturbation_setting, {})
    if not params:
        return f"{cls.label} ({perturbation_setting})"
    rendered = ", ".join(f"{key}={value}" for key, value in params.items())
    return f"{cls.label} ({rendered})"


def family_of(perturbation_type: str) -> Optional[str]:
    """Family constant for a branch, or None for ORIGINAL / NO_AUDIO."""
    cls = PERTURBATION_BY_NAME.get(perturbation_type.upper())
    return cls.family if cls is not None else None


def library_summary() -> Dict[str, Dict[str, int]]:
    """Per-family class and setting counts, in the paper's family order."""
    summary: Dict[str, Dict[str, int]] = {}
    for family in FAMILY_ORDER:
        members = [cls for cls in ALL_CLASSES if cls.family == family]
        summary[family] = {
            "classes": len(members),
            "settings": sum(len(cls.settings) for cls in members),
            "stochastic_classes": sum(1 for cls in members if cls.stochastic),
        }
    return summary


def _self_check() -> None:
    """Verify the registry against the counts the paper reports."""
    n_settings = sum(len(cls.settings) for cls in ALL_CLASSES)
    n_perturbations = n_settings + 1  # + NO_AUDIO
    n_types = len(ALL_CLASSES) + len(BASELINE_PERTURBATIONS)
    summary = library_summary()

    print(f"{'Family':<22}{'classes':>9}{'settings':>10}{'stochastic':>12}")
    print("-" * 53)
    for family in FAMILY_ORDER:
        counts = summary[family]
        print(f"{FAMILY_LABELS[family]:<22}{counts['classes']:>9}"
              f"{counts['settings']:>10}{counts['stochastic_classes']:>12}")
    print("-" * 53)
    print(f"{'Total':<22}{len(ALL_CLASSES):>9}{n_settings:>10}{len(STOCHASTIC_TYPES):>12}")
    print()
    print(f"{n_perturbations} perturbations across {n_types} branch types "
          f"(paper reports 105 across 38)")

    assert len(ALL_CLASSES) == len(PERTURBATION_BY_NAME), "duplicate perturbation name in ALL_CLASSES"
    assert n_perturbations == 105, f"expected 105 perturbations, got {n_perturbations}"
    assert n_types == 38, f"expected 38 types, got {n_types}"
    for cls in ALL_CLASSES:
        assert cls.family in FAMILY_ORDER, f"{cls.name} has unknown family {cls.family!r}"
        assert cls.settings, f"{cls.name} declares no settings"
    configs = list(iter_perturbation_configs())
    assert len(configs) == n_settings + len(BASELINE_PERTURBATIONS), "config iterator lost branches"
    assert len(set(configs)) == len(configs), "config iterator yielded a duplicate"
    print("OK: registry matches the paper's reported counts.")


if __name__ == "__main__":
    _self_check()
