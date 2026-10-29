"""Envelope cross-correlation: is the time base consistent?

A contrast below one (Tab. 5) has a mundane alternative explanation: if the
generated audio were shifted by more than the +-300 ms ETA window -- a clip
trimmed differently, a resampling latency, a padding convention -- the
instrument would report no response at the reference onsets even for a
perfectly aligned model.

This module rules that out. It cross-correlates the generated envelope against
the reference envelope over +-300 ms and reports (a) the best lag and (b) the
correlation at that lag. A best lag well inside the window means the time base
is consistent; a low correlation *at that lag* then says the envelope itself
does not track the reference, which is a statement about the model rather than
about the measurement.

Both a spectral-flux and an energy (RMS) envelope are used, since they fail in
different ways.

Usage
-----
    python -m eval.envelope_xcorr --gen_dir path/to/audio --split gh
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

MAX_LAG = 30      # +-300 ms at 10 ms frames


def _load(path):
    y, _ = librosa.load(path, sr=C.SR, mono=True)
    n = int(C.SR * C.DURATION)
    return np.pad(y[:n], (0, max(0, n - len(y[:n]))))


def _xcorr(a, b):
    """Normalised cross-correlation. Positive lag = a later than b."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    best, best_lag = -2.0, 0
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        if lag >= 0:
            u, v = a[lag:], (b[: n - lag] if lag else b)
        else:
            u, v = a[: n + lag], b[-lag:]
        m = min(len(u), len(v))
        if m < 20:
            continue
        c = float(np.dot(u[:m], v[:m]) / m)
        if c > best:
            best, best_lag = c, lag
    return best, best_lag * (C.HOP / C.SR) * 1000.0


def _one(args):
    gen_path, ref_path = args
    try:
        g, r = _load(gen_path), _load(ref_path)
        go = librosa.onset.onset_strength(y=g, sr=C.SR, hop_length=C.HOP)
        ro = librosa.onset.onset_strength(y=r, sr=C.SR, hop_length=C.HOP)
        gr = librosa.feature.rms(y=g, hop_length=C.HOP)[0]
        rr = librosa.feature.rms(y=r, hop_length=C.HOP)[0]
        po, lo = _xcorr(go, ro)
        pr, lr = _xcorr(gr, rr)
        return po, lo, pr, lr
    except Exception as exc:
        raise RuntimeError(f"Could not correlate audio: {gen_path}, {ref_path}") from exc


def measure(gen_dir, ref_dir, ids, gen_suffix=".flac", ref_suffix=".wav", workers=12):
    gen_dir, ref_dir = Path(gen_dir), Path(ref_dir)
    from validation import validate_audio
    validate_audio(gen_dir, ids, gen_suffix, reject_extra=False)
    validate_audio(ref_dir, ids, ref_suffix, reject_extra=False)
    jobs = [(gen_dir / f"{v}{gen_suffix}", ref_dir / f"{v}{ref_suffix}") for v in ids
            if (gen_dir / f"{v}{gen_suffix}").exists()
            and (ref_dir / f"{v}{ref_suffix}").exists()]
    if not jobs:
        raise FileNotFoundError("no matching generated/reference pairs")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = [r for r in ex.map(_one, jobs, chunksize=4) if r]
    a = np.array(res)
    return {"n": len(res),
            "xcorr_peak_flux": round(float(a[:, 0].mean()), 4),
            "best_lag_flux_ms": round(float(np.median(a[:, 1])), 1),
            "xcorr_peak_rms": round(float(a[:, 2].mean()), 4),
            "best_lag_rms_ms": round(float(np.median(a[:, 3])), 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--split", choices=["gh", "wild"], default="gh")
    ap.add_argument("--suffix", default=".flac")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    onsets, inventory = C.preflight(a.split, a.gen_dir, a.suffix)
    ref_dir = (C.GH_ROOT if a.split == "gh" else C.VGG_ROOT) / "audio"
    C.preflight(a.split, ref_dir, ".wav")
    r = measure(a.gen_dir, ref_dir, list(onsets), a.suffix)
    r["input_inventory"] = inventory
    if a.out:
        from validation import save_json
        save_json(a.out, r)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
