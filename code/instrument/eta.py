"""Event-triggered average (ETA) — the detector-free instrument of Sec. 3.2.

The onset scores in the paper come from a peak-picking detector, which raises
the objection that a falling F1 could reflect the detector losing calibration
rather than the model losing alignment. This module removes the detector: it
never picks peaks, never thresholds and never applies non-maximum suppression.

For every reference onset t it takes the +-300 ms window of the generated
onset-strength envelope centred at t and averages those windows over the whole
test split. The same average is computed at *control* anchors -- random times
at least 500 ms from any reference onset. Three quantities follow:

    prominence          peak of the event curve minus its window-edge baseline
    control_prominence  the same for the control curve
    contrast            prominence / control_prominence

`contrast` is the quantity Tab. 5 reports. It is a ratio, so a global gain on
the waveform leaves it unchanged, and a value below one means the envelope does
not respond more strongly at annotated onsets than away from them.

Two envelope variants are computed side by side:

    osz     spectral flux (librosa onset strength), clip-wise z-normalised
    rawosz  the same flux without normalisation

The unnormalised variant is a control: clip-wise z-normalisation cannot change
a ratio within a clip, but it does change how clips are weighted when their
curves are averaged, and `rawosz` removes that degree of freedom.

Usage
-----
    python -m instrument.eta --gen_dir path/to/audio --split gh --out eta.json
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


def _envelopes(y):
    raw = librosa.onset.onset_strength(y=y, sr=C.SR, hop_length=C.HOP)
    z = (raw - raw.mean()) / (raw.std() + 1e-9)
    return z, raw


def _load(path):
    y, _ = librosa.load(path, sr=C.SR, mono=True)
    n = int(C.SR * C.DURATION)
    return np.pad(y[:n], (0, max(0, n - len(y[:n]))))


def _one_clip(args):
    """Average the event and control windows for a single clip."""
    path, onsets = args
    try:
        envs = _envelopes(_load(path))
        n_frames = len(envs[0])
        rng = np.random.default_rng(0)          # fixed: control anchors are reproducible
        gate = int(C.CONTROL_MIN_MS / 1000 * C.SR / C.HOP)
        event = [[], []]
        control = [[], []]
        for t in onsets:
            centre = int(round(t * C.SR / C.HOP))
            if centre - C.ETA_HALF < 0 or centre + C.ETA_HALF + 1 > n_frames:
                continue
            for i, e in enumerate(envs):
                event[i].append(e[centre - C.ETA_HALF:centre + C.ETA_HALF + 1])
            for _ in range(10):                 # rejection-sample one control anchor
                c = int(rng.integers(C.ETA_HALF, n_frames - C.ETA_HALF - 1))
                if all(abs(c - int(round(u * C.SR / C.HOP))) > gate for u in onsets):
                    for i, e in enumerate(envs):
                        control[i].append(e[c - C.ETA_HALF:c + C.ETA_HALF + 1])
                    break
        if not event[0]:
            return None
        return ([np.mean(x, 0) for x in event],
                [np.mean(x, 0) if x else None for x in control],
                len(event[0]))
    except Exception as exc:
        raise RuntimeError(f"Could not measure audio: {path}") from exc


def curve_stats(curve):
    """Peak height above the window-edge baseline, its lag, and the half-width."""
    edge = np.concatenate([curve[:C.ETA_EDGE], curve[-C.ETA_EDGE:]])
    baseline = float(np.mean(edge))
    i = int(np.argmax(curve))
    peak = float(curve[i])
    prom = peak - baseline
    half = baseline + prom / 2.0
    lo = i
    while lo > 0 and curve[lo] > half:
        lo -= 1
    hi = i
    while hi < len(curve) - 1 and curve[hi] > half:
        hi += 1
    ms = lambda k: k * (C.HOP / C.SR) * 1000
    return {"prominence": round(prom, 6), "baseline": round(baseline, 6),
            "lag_ms": round(ms(i - C.ETA_HALF), 1), "fwhm_ms": round(ms(hi - lo), 1)}


def measure(audio_dir, onsets, suffix=".flac", workers=12, keep_curves=False):
    """Run the instrument over one directory of generated (or reference) audio."""
    audio_dir = Path(audio_dir)
    from validation import validate_audio
    validate_audio(audio_dir, onsets, suffix, reject_extra=False)
    jobs = [(audio_dir / f"{v}{suffix}", ts) for v, ts in onsets.items()
            if (audio_dir / f"{v}{suffix}").exists()]
    if not jobs:
        raise FileNotFoundError(f"no audio matching *{suffix} in {audio_dir}")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = [r for r in ex.map(_one_clip, jobs, chunksize=4) if r]
    out = {"n_clips": len(res), "n_onsets": int(sum(r[2] for r in res))}
    for i, tag in enumerate(("osz", "rawosz")):
        ev = np.mean([r[0][i] for r in res], 0)
        cs = [r[1][i] for r in res if r[1][i] is not None]
        se = curve_stats(ev)
        entry = {"prominence": se["prominence"], "lag_ms": se["lag_ms"],
                 "fwhm_ms": se["fwhm_ms"]}
        if cs:
            cc = np.mean(cs, 0)
            sc = curve_stats(cc)
            entry["control_prominence"] = sc["prominence"]
            entry["contrast"] = round(se["prominence"] / max(sc["prominence"], 1e-9), 3)
            if keep_curves:
                entry["curve_control"] = [round(float(x), 5) for x in cc]
        if keep_curves:
            entry["curve_event"] = [round(float(x), 5) for x in ev]
        out[tag] = entry
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gen_dir", required=True,
                    help="directory of audio to measure (generated or reference)")
    ap.add_argument("--split", choices=["gh", "wild"], default="gh")
    ap.add_argument("--suffix", default=".flac", help="audio file extension")
    ap.add_argument("--out", default=None, help="write JSON here")
    ap.add_argument("--keep_curves", action="store_true",
                    help="store the averaged +-300 ms waveforms (for plotting)")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    onsets, inventory = C.preflight(a.split, a.gen_dir, a.suffix)
    r = measure(a.gen_dir, onsets, a.suffix, a.workers, a.keep_curves)
    r["input_inventory"] = inventory
    txt = json.dumps(r, indent=1)
    if a.out:
        from validation import save_json
        save_json(a.out, r)
    print(f"clips {r['n_clips']}  onsets {r['n_onsets']}")
    for tag in ("osz", "rawosz"):
        e = r[tag]
        print(f"  {tag:7s} prominence {e['prominence']:.4f} · "
              f"control {e.get('control_prominence', float('nan')):.4f} · "
              f"contrast {e.get('contrast', float('nan')):.3f} · "
              f"lag {e['lag_ms']:+.0f} ms")


if __name__ == "__main__":
    main()
