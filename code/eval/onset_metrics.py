"""Onset scoring: event F1 at several tolerances, placement bias, attack rise.

Two scores appear in the paper and they are not interchangeable.

frame-level F1
    The protocol prior V2A work tabulates. In its original implementation the
    peak-picking `delta` *also* sets the matching window, so delta=0.3 carries a
    300 ms tolerance: the sensitivity of the detector and the tolerance of the
    match cannot be varied independently. Tab. 3-5 of the paper use this score
    because it is what the literature reports.

event F1 at {50, 100, 300} ms
    Computed here. The peak-picking delta and the matching tolerance are
    separate arguments, which is what makes the tolerance decomposition of
    Sec. 3.1 possible: the same detected peaks are scored at three tolerances.

Matching is greedy nearest-neighbour, one predicted onset per reference onset.

Usage
-----
    python -m eval.onset_metrics --gen_dir path/to/audio --split gh
"""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


# ---------------------------------------------------------------------------
# Detection and non-maximum suppression.
#
# These two functions are the reference implementation used by prior V2A work
# (originally from CondFoleyGen's predict_onset.py) and are reproduced here
# unchanged, including two details that matter for exact comparability:
#
#   1. audio is loaded at 22.05 kHz, not at the 16 kHz the models generate at;
#   2. `onset_nms` mutates `onset_remain` while iterating over it, so the
#      suppression is not exhaustive.
#
# Both are properties of the published protocol rather than choices of ours.
# Re-implementing them "correctly" changes every reported onset number, so the
# release keeps them as they are; the tolerance decomposition of Sec. 3.1 is
# unaffected because it varies only the matching tolerance on the same peaks.
# ---------------------------------------------------------------------------
DETECT_SR = 22050
CONF_INTERVAL = int(0.05 * DETECT_SR)


def detect_onsets(path, delta):
    """librosa peak-picking at the protocol's sample rate. Returns samples."""
    wav, _ = librosa.load(path, sr=DETECT_SR)
    wav = wav[: int(DETECT_SR * C.DURATION)]
    onsets = librosa.onset.onset_detect(y=wav, sr=DETECT_SR, units="samples",
                                        delta=delta)
    wav_norm = (wav - wav.min()) / (wav.max() - wav.min() + 1e-6)
    return onsets, wav_norm


def onset_nms(onsets, wav_norm, window=0.05):
    """Reference NMS. See the note above before changing anything here."""
    confidence = [np.max(wav_norm[o - CONF_INTERVAL:o + CONF_INTERVAL]) for o in onsets]
    onset_remain = onsets.tolist()
    output = []
    for idx in np.argsort(confidence)[::-1]:
        cur = onsets[idx]
        if cur not in onset_remain:
            continue
        output.append(cur)
        onset_remain.remove(cur)
        for o in onset_remain:
            if abs(cur - o) < window * DETECT_SR:
                onset_remain.remove(o)
    return np.array(sorted(output))


def score(audio_dir, onsets, delta, suffix=".flac", rise=True):
    audio_dir = Path(audio_dir)
    stats = {t: [0, 0, 0] for t in C.TOLERANCES}      # tp, n_pred, n_ref
    errors, rises, used = [], [], 0
    for vid, refs in onsets.items():
        f = audio_dir / f"{vid}{suffix}"
        if not f.is_file():
            raise FileNotFoundError(f)
        if not refs:
            continue
        onsets, wav_norm = detect_onsets(f, delta)
        onsets = onset_nms(onsets, wav_norm) if len(onsets) else np.array([])
        pred = onsets / DETECT_SR                    # samples -> seconds
        used += 1
        for tol in C.TOLERANCES:
            remaining = pred.tolist()
            tp = 0
            for g in refs:
                cand = [p for p in remaining if abs(p - g) < tol]
                if cand:
                    p = min(cand, key=lambda x: abs(x - g))
                    remaining.remove(p)
                    tp += 1
                    if tol == 0.30:
                        errors.append(p - g)          # signed placement error
            stats[tol][0] += tp
            stats[tol][1] += len(pred)
            stats[tol][2] += len(refs)
        if rise:
            rises.extend(attack_rise(f, refs))
    out = {"n_clips": used}
    for tol in C.TOLERANCES:
        tp, npred, nref = stats[tol]
        P, R = tp / max(npred, 1), tp / max(nref, 1)
        k = int(tol * 1000)
        out[f"f1@{k}"] = round(2 * P * R / max(P + R, 1e-9), 4)
        out[f"precision@{k}"] = round(P, 4)
        out[f"recall@{k}"] = round(R, 4)
    if errors:
        e = np.array(errors)
        out["bias_ms"] = round(float(e.mean() * 1000), 1)
        out["abs_err_median_ms"] = round(float(np.median(np.abs(e)) * 1000), 1)
    if rises:
        out["rise_median_ms"] = round(float(np.median(rises)), 2)
    return out


def attack_rise(path, refs):
    """10%-to-90% rise of the waveform envelope in a window around each onset."""
    w, _ = librosa.load(path, sr=C.SR, mono=True)
    n = int(C.SR * C.DURATION)
    w = np.pad(w[:n], (0, max(0, n - len(w[:n]))))
    out = []
    for g in refs:
        s, e = max(0, int((g - .05) * C.SR)), min(len(w), int((g + .25) * C.SR))
        seg = np.abs(w[s:e])
        if len(seg) < 10 or seg.max() < 1e-4:
            continue
        i90 = int(np.argmax(seg >= .9 * seg.max()))
        i10 = int(np.argmax(seg[:i90 + 1] >= .1 * seg.max())) if i90 > 0 else 0
        out.append((i90 - i10) / C.SR * 1000)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--split", choices=["gh", "wild"], default="gh")
    ap.add_argument("--suffix", default=".flac")
    ap.add_argument("--delta", type=float, default=None,
                    help="peak-picking delta; defaults to the split's value")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    onsets, inventory = C.preflight(a.split, a.gen_dir, a.suffix)
    default_delta = C.PEAK_DELTA_GH if a.split == "gh" else C.PEAK_DELTA_WILD
    delta = default_delta if a.delta is None else a.delta
    r = score(a.gen_dir, onsets, delta, a.suffix)
    r["input_inventory"] = inventory
    r["peak_delta"] = delta
    if a.out:
        from validation import save_json
        save_json(a.out, r)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
