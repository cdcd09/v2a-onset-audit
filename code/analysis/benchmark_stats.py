"""Class distribution and onset statistics of the in-the-wild benchmark (Sec. 5.1).

Reports what the main text promises the supplement will contain: how many
classes the 266 clips span, how onsets are distributed over clips, and why
scoring uses 263 of them.

    python -m analysis.benchmark_stats --out outputs/benchmark_stats.json
"""
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=None, help="benchmark CSV (defaults to config)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    path = Path(a.csv) if a.csv else C.VGG_ROOT / "benchmark_266.csv"
    from validation import load_benchmark
    load_benchmark(path, C.DURATION)
    rows = list(csv.DictReader(open(path)))
    onsets = {r["video_id"]: [float(x) for x in r["onset_times"].split(",")]
              for r in rows if r.get("onset_times")}
    labels = {r["video_id"]: r["query_label"] for r in rows}

    total = sum(len(v) for v in onsets.values())
    kept = {k: [t for t in v if t < C.DURATION] for k, v in onsets.items()}
    scored = {k: v for k, v in kept.items() if v}
    dropped = {k: onsets[k] for k in kept if not kept[k]}
    counts = sorted(len(v) for v in onsets.values())
    cls = Counter(labels.values())

    out = {
        "n_clips": len(rows),
        "n_onsets_total": total,
        "n_classes": len(cls),
        "onsets_per_clip": {
            "mean": round(total / len(onsets), 2),
            "median": counts[len(counts) // 2],
            "min": counts[0], "max": counts[-1]},
        "scoring_window_s": C.DURATION,
        "n_clips_scored": len(scored),
        "n_onsets_scored": sum(len(v) for v in scored.values()),
        "clips_dropped": {k: {"onsets": v, "label": labels[k]} for k, v in dropped.items()},
        "top_classes": cls.most_common(a.top),
        "n_classes_beyond_top": len(cls) - min(a.top, len(cls)),
    }
    if a.out:
        from validation import save_json
        save_json(a.out, out)

    print(f"clips {out['n_clips']} · onsets {out['n_onsets_total']} · "
          f"classes {out['n_classes']}")
    o = out["onsets_per_clip"]
    print(f"onsets per clip: mean {o['mean']} · median {o['median']} · "
          f"range {o['min']}-{o['max']}")
    print(f"scored within {C.DURATION:.0f} s: {out['n_clips_scored']} clips, "
          f"{out['n_onsets_scored']} onsets")
    for k, v in out["clips_dropped"].items():
        print(f"  dropped {k}: onsets {v['onsets']} ({v['label']})")
    print(f"top {a.top} classes (+{out['n_classes_beyond_top']} more):")
    for name, n in out["top_classes"]:
        print(f"  {n:3d}  {name}")


if __name__ == "__main__":
    main()
