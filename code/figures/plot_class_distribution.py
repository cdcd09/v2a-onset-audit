"""Class composition of the in-the-wild benchmark (Sec. S1.2).

Reads the JSON written by `analysis.benchmark_stats` and draws the head of the
class distribution plus a single bar for the tail.

    python -m figures.plot_class_distribution \
        --stats outputs/benchmark_stats.json --out outputs/class_distribution.pdf
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, GREY = "#1f4e79", "#8c8c8c"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", default="fig_class_distribution.pdf")
    a = ap.parse_args()

    d = json.load(open(a.stats))
    top = d["top_classes"]
    n_tail_cls = d["n_classes_beyond_top"]
    n_tail_clips = d["n_clips"] - sum(n for _, n in top)

    names = [c for c, _ in top][::-1] + [f"{n_tail_cls} further classes"]
    vals = [n for _, n in top][::-1] + [n_tail_clips]
    colors = [BLUE] * len(top) + [GREY]
    y = np.arange(len(vals))

    fig, ax = plt.subplots(figsize=(4.3, 3.4))
    ax.barh(y, vals, color=colors, height=.72)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.get_yticklabels()[-1].set_style("italic")
    ax.get_yticklabels()[-1].set_color(GREY)
    for yi, v in zip(y, vals):
        ax.text(v + 1.5, yi, str(v), va="center", fontsize=7,
                color=GREY if yi == y[-1] else BLUE)
    ax.set_xlabel("clips", fontsize=8)
    ax.set_xlim(0, max(vals) * 1.14)
    ax.set_title(f"{d['n_clips']} clips over {d['n_classes']} VGGSound classes",
                 fontsize=9, loc="left")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}  (head {sum(n for _,n in top)} clips, "
          f"tail {n_tail_clips} clips in {n_tail_cls} classes)")


if __name__ == "__main__":
    main()
