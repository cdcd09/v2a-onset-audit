"""Additive-background control for the ETA instrument (Sec. 3.2, supplement).

The contrast ratio could in principle be reporting how dense and loud a model's
output is rather than how well it aligns: the reference recordings are quiet
studio impacts and score 113, while open-domain models that fill the space
between events score low. This control measures contrast at matched control
response, while retaining the instrument's alignment and background components.

We add background *to the reference audio* until its control-anchor response
matches a model's, then measure contrast again. If the reference still exceeds
the model at matched background, background level alone does not explain the deficit.

Three design choices make the control tight:

  waveform domain     noise is added to the audio and the envelope recomputed
                      from scratch, so the non-linearity of the representation
                      is included; adding to the envelope would not test it
  the model's own     background is cut from the model's generated audio at
  background          least 500 ms from any reference onset and tiled with a
                      5 ms crossfade; clips with less than one second of eligible
                      background fall back to the whole generated waveform
  one global gain     a single scalar for the whole split, bisected to hit the
                      target within 2%; per-clip gains would add a degree of
                      freedom that could be tuned

A white-noise condition is available as a null: it answers whether *any* added
energy disturbs the measure, or specifically the structure the model produces.

A uniform positive term added equally to E and B moves (E+b)/(B+b) toward
one and cannot turn E/B > 1 into a ratio below one. This algebraic observation
does not describe every structured waveform mixture.

Usage
-----
    python -m instrument.background_control --noise_dir path/to/model_audio \\
        --targets 0.089 0.143 0.191 --out bg_control.json
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
from instrument.eta import curve_stats

_NOISE_DIR = None
_NOISE_KIND = "model"


def _load(path, sr=None):
    y, _ = librosa.load(path, sr=sr or C.SR, mono=True)
    n = int(C.SR * C.DURATION)
    return np.pad(y[:n], (0, max(0, n - len(y[:n]))))


def _model_background(gen, onsets):
    """Tile the parts of `gen` that lie >=500 ms from any reference onset."""
    n = len(gen)
    keep = np.ones(n, bool)
    half = int(C.CONTROL_MIN_MS / 1000 * C.SR)
    for t in onsets:
        c = int(t * C.SR)
        keep[max(0, c - half):min(n, c + half)] = False
    seg = gen[keep]
    if len(seg) < C.SR:                       # too little background: use it all
        seg = gen
    tiled = np.tile(seg, int(np.ceil(n / len(seg))))[:n]
    w = int(0.005 * C.SR)                     # 5 ms crossfade at each seam
    for b in range(len(seg), n, len(seg)):
        if b + w > n:
            break
        f = np.linspace(0, 1, w)
        tiled[b:b + w] = tiled[b:b + w] * f + tiled[b - w:b] * (1 - f)
    return tiled


def _one_clip(args):
    vid, onsets, gain, suffix = args
    try:
        ref = _load(C.GH_ROOT / "audio" / f"{vid}.wav")
        if gain > 0:
            if _NOISE_KIND == "white":
                rng = np.random.default_rng(abs(hash(vid)) % (2 ** 32))
                noise = rng.normal(0, 1.0, len(ref)).astype(np.float32)
            else:
                p = Path(_NOISE_DIR) / f"{vid}{suffix}"
                if not p.is_file():
                    raise FileNotFoundError(p)
                noise = _model_background(_load(p), onsets)
            y = ref + gain * noise
            peak = np.abs(y).max()
            if peak > 1.0:
                y = y / peak                  # a global scale cannot change a ratio
        else:
            y = ref
        raw = librosa.onset.onset_strength(y=y, sr=C.SR, hop_length=C.HOP)
        envs = ((raw - raw.mean()) / (raw.std() + 1e-9), raw)
        n_fr = len(raw)
        rng = np.random.default_rng(0)
        gate = int(C.CONTROL_MIN_MS / 1000 * C.SR / C.HOP)
        ev, ct = [[], []], [[], []]
        for t in onsets:
            c = int(round(t * C.SR / C.HOP))
            if c - C.ETA_HALF < 0 or c + C.ETA_HALF + 1 > n_fr:
                continue
            for i, e in enumerate(envs):
                ev[i].append(e[c - C.ETA_HALF:c + C.ETA_HALF + 1])
            for _ in range(10):
                rc = int(rng.integers(C.ETA_HALF, n_fr - C.ETA_HALF - 1))
                if all(abs(rc - int(round(u * C.SR / C.HOP))) > gate for u in onsets):
                    for i, e in enumerate(envs):
                        ct[i].append(e[rc - C.ETA_HALF:rc + C.ETA_HALF + 1])
                    break
        if not ev[0]:
            return None
        return ([np.mean(x, 0) for x in ev],
                [np.mean(x, 0) if x else None for x in ct], len(ev[0]))
    except Exception as exc:
        raise RuntimeError(f"Could not measure background control: {vid}") from exc


def _init_worker(noise_dir, noise_kind):
    global _NOISE_DIR, _NOISE_KIND
    _NOISE_DIR, _NOISE_KIND = noise_dir, noise_kind


def measure(gain, onsets, suffix, workers=12):
    from validation import validate_audio
    validate_audio(C.GH_ROOT / "audio", onsets, ".wav", reject_extra=False)
    if _NOISE_KIND == "model":
        validate_audio(_NOISE_DIR, onsets, suffix, reject_extra=False)
    jobs = [(v, ts, gain, suffix) for v, ts in onsets.items()]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(_NOISE_DIR, _NOISE_KIND)) as ex:
        res = [r for r in ex.map(_one_clip, jobs, chunksize=4) if r]
    out = {"n_clips": len(res), "n_onsets": int(sum(r[2] for r in res))}
    for i, tag in enumerate(("osz", "rawosz")):
        ev = np.mean([r[0][i] for r in res], 0)
        cs = [r[1][i] for r in res if r[1][i] is not None]
        se, sc = curve_stats(ev), curve_stats(np.mean(cs, 0))
        out[tag] = {"prominence": se["prominence"],
                    "control_prominence": sc["prominence"],
                    "contrast": round(se["prominence"] / max(sc["prominence"], 1e-9), 3),
                    "lag_ms": se["lag_ms"]}
    return out


def bisect_to(target, onsets, suffix, tol=0.02, max_it=14, workers=12):
    """Find the global gain whose control response lands within `tol` of target."""
    lo, hi, it = 0.0, 1.0, 0
    while True:
        r = measure(hi, onsets, suffix, workers); it += 1
        if r["osz"]["control_prominence"] >= target or hi > 400:
            break
        lo, hi = hi, hi * 2
    best = None
    for _ in range(max_it):
        mid = (lo + hi) / 2
        r = measure(mid, onsets, suffix, workers); it += 1
        got = r["osz"]["control_prominence"]
        if best is None or abs(got - target) < abs(best[1] - target):
            best = (mid, got, r)
        if abs(got - target) / target <= tol:
            break
        lo, hi = (mid, hi) if got < target else (lo, mid)
    return best[0], best[2], it


def main():
    global _NOISE_DIR, _NOISE_KIND
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--noise_dir", help="generated audio to draw background from")
    ap.add_argument("--noise", choices=["model", "white"], default="model")
    ap.add_argument("--suffix", default=".flac")
    ap.add_argument("--targets", type=float, nargs="+", required=True,
                    help="control-response levels to match, e.g. 0.089 0.191")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    _NOISE_DIR, _NOISE_KIND = a.noise_dir, a.noise
    if a.noise == "model" and not a.noise_dir:
        ap.error("--noise_dir is required for --noise model")

    onsets, inventory = C.preflight("gh", C.GH_ROOT / "audio", ".wav")
    if a.noise == "model":
        C.preflight("gh", a.noise_dir, a.suffix)
    res = {"meta": {"noise": a.noise, "n_clips": len(onsets),
                    "input_inventory": inventory}, "conditions": []}

    zero = measure(0.0, onsets, a.suffix, a.workers)
    res["meta"]["unmixed_reference"] = zero
    print(f"[reference]  contrast {zero['osz']['contrast']:.1f} · "
          f"control {zero['osz']['control_prominence']:.4f} · "
          f"lag {zero['osz']['lag_ms']:+.0f} ms", flush=True)

    for t in a.targets:
        gain, r, it = bisect_to(t, onsets, a.suffix, workers=a.workers)
        got = r["osz"]["control_prominence"]
        res["conditions"].append({
            "target_control": t, "gain": round(gain, 4),
            "achieved_control": got, "target_reached": abs(got - t) / t <= 0.05,
            "iterations": it, "osz": r["osz"], "rawosz": r["rawosz"]})
        print(f"[target {t:.3f}] gain {gain:.3f} -> control {got:.4f} · "
              f"contrast {r['osz']['contrast']:.2f} (raw {r['rawosz']['contrast']:.2f})"
              f"{'' if abs(got-t)/t <= 0.05 else '  [not reached: saturated]'}", flush=True)

    if a.out:
        from validation import save_json
        save_json(a.out, res)


if __name__ == "__main__":
    main()
