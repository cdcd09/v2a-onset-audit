"""Reproduce the two-endpoint detector-threshold audit with the released scorer."""
from __future__ import annotations

import os
# CPU-only, at most two processes. Set limits before importing numerical packages.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import config as C
from eval import onset_metrics
from validation import save_json

EXPECTED = {
    10000: {"f1@50": 0.1808, "f1@100": 0.4579, "f1@300": 0.7946},
    150000: {"f1@50": 0.2079, "f1@100": 0.4987, "f1@300": 0.7640},
}


def one_job(job):
    step, directory, suffix, delta, references = job
    result = onset_metrics.score(directory, references, delta, suffix=suffix, rise=False)
    if result["n_clips"] != len(references):
        raise RuntimeError("Scored clip count differs from validated eligible windows")
    return {"step": step, "delta": delta, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early-dir", type=Path,
                        default=C.GEN_ROOT / "vanilla_10000" / "gh")
    parser.add_argument("--late-dir", type=Path,
                        default=C.GEN_ROOT / "vanilla_150000" / "gh")
    parser.add_argument("--suffix", default=".flac")
    parser.add_argument("--deltas", type=float, nargs="+", default=[.15, .20, .30, .40, .50])
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--out", type=Path, default=Path("outputs/delta_sweep.json"))
    parser.add_argument("--check-reference", action="store_true",
                        help="Check all six published endpoint F1 values at delta=.3")
    parser.add_argument("--reference-atol", type=float, default=0.0,
                        help="Absolute F1 tolerance for reference checks (default: exact rounded agreement)")
    args = parser.parse_args()
    if not 0 <= args.reference_atol < float("inf"):
        parser.error("reference-atol must be finite and nonnegative")
    if len(args.deltas) != len(set(args.deltas)) or any(d < 0 for d in args.deltas):
        parser.error("deltas must be unique nonnegative values")
    if args.check_reference and .3 not in args.deltas:
        parser.error("--check-reference requires delta .3")
    inventories, jobs = {}, []
    for step, directory in [(10000, args.early_dir), (150000, args.late_dir)]:
        refs, inventory = C.preflight("gh", directory, args.suffix)
        inventories[str(step)] = inventory
        jobs += [(step, str(directory), args.suffix, delta, refs) for delta in args.deltas]
    if args.workers == 1:
        rows = [one_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            rows = list(pool.map(one_job, jobs))
    rows.sort(key=lambda r: (r["delta"], r["step"]))
    checks = [{"step": r["step"], "expected": EXPECTED[r["step"]],
               "observed": {k: r[k] for k in EXPECTED[r["step"]]},
               "exact_matches": all(r[k] == v for k, v in EXPECTED[r["step"]].items()),
               "absolute_tolerance": args.reference_atol,
               "matches": all(abs(r[k] - v) <= args.reference_atol + 1e-12
                              for k, v in EXPECTED[r["step"]].items())}
              for r in rows if r["delta"] == .3]
    source_root = Path(__file__).resolve().parents[1]
    hashes = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [Path(__file__), source_root / "eval/onset_metrics.py",
                        source_root / "config.py", source_root / "validation.py"]}
    report = {
        "scope": "Greatest Hits unmodified 10K and 150K endpoints only; not full-trajectory or perceptual validation",
        "protocol": {"detection_sr_hz": 22050, "duration_s": 8.0,
                     "matching_tolerances_s": [.05, .10, .30],
                     "matching_boundary": "strict abs(pred-ref) < tolerance",
                     "legacy_non_exhaustive_nms_preserved": True, "rise": False},
        "input_inventory": inventories,
        "environment": {"python": platform.python_version(),
                        "packages": {k: importlib.metadata.version(k) for k in
                                     ["numpy", "scipy", "librosa", "soundfile", "numba", "soxr"]}},
        "source_sha256": hashes, "results": rows, "reference_checks": checks,
    }
    save_json(args.out, report)
    with args.out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    if args.check_reference and not (len(checks) == 2 and all(c["matches"] for c in checks)):
        raise RuntimeError("Reference reproduction failed; inspect saved JSON")
    print(json.dumps({"runs": len(rows), "reference_checks": checks,
                      "manifest_clips": inventories["10000"]["manifest_clips"],
                      "eligible_windows": inventories["10000"]["eligible_windows"]}, indent=2))


if __name__ == "__main__":
    main()
