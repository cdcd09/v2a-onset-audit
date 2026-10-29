"""Two views of the agreement between the independent annotation lanes (Sec. S1.4).

Left: the signed discrepancy between matched onsets, which is what rules out a
systematic offset between annotators. Right: per-clip onset counts, which is
what shows the two lanes segment the clips the same way.

    python figures/plot_annotation_agreement.py \
        --csv /path/to/downloaded-release/metadata/annotation_lanes.csv \
        --out outputs/annotation_agreement.pdf
"""
import argparse, csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, RED = "#1f4e79", "#b04a2a"


def parse(f):
    return sorted(float(x) for x in f.strip().split(";") if x)


def greedy(ref, est, tol):
    if not ref or not est:
        return []
    pairs = sorted((abs(r - e), i, j, e - r)
                   for i, r in enumerate(ref) for j, e in enumerate(est)
                   if abs(r - e) <= tol)
    ur, ue, out = set(), set(), []
    for _, i, j, sd in pairs:
        if i in ur or j in ue:
            continue
        ur.add(i); ue.add(j); out.append(sd)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="fig_annotation_agreement.pdf")
    a = ap.parse_args()

    ons, use = {}, {}
    for r in csv.DictReader(open(a.csv)):
        ons.setdefault(r["clip_id"], {})[r["source"]] = parse(r["onsets_sec"])
        use.setdefault(r["clip_id"], {})[r["source"]] = r["usable"] == "1"
    need = {"gt_release", "lane_A", "lane_B"}
    both = sorted(c for c in ons if need <= set(ons[c])
                  and use[c]["lane_A"] and use[c]["lane_B"])

    dt = np.array([d * 1e3 for c in both
                   for d in greedy(ons[c]["lane_A"], ons[c]["lane_B"], 0.30)])
    na = np.array([len(ons[c]["lane_A"]) for c in both])
    nb = np.array([len(ons[c]["lane_B"]) for c in both])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.5))

    ax1.hist(dt, bins=np.arange(-100, 101, 5), color=BLUE, alpha=.85, lw=0)
    ax1.axvline(0, color="0.55", lw=.8, zorder=0)
    ax1.axvline(np.median(dt), color=RED, lw=1.4, ls="--",
                label=f"median {np.median(dt):+.1f} ms")
    ax1.set_xlabel("lane B $-$ lane A, matched onsets (ms)", fontsize=8)
    ax1.set_ylabel("matched pairs", fontsize=8)
    ax1.legend(fontsize=7, frameon=False)
    ax1.set_title(f"no offset between annotators  (sd {dt.std():.1f} ms)",
                  fontsize=9)

    hi = max(na.max(), nb.max()) + 1
    ax2.plot([0, hi], [0, hi], color="0.7", lw=.8, zorder=0)
    ax2.scatter(na, nb, s=13, color=BLUE, alpha=.55, lw=0)
    ax2.set_xlim(0, hi); ax2.set_ylim(0, hi)
    ax2.set_xlabel("onsets marked, lane A", fontsize=8)
    ax2.set_ylabel("lane B", fontsize=8)
    ax2.set_title(f"per-clip counts  ($r={np.corrcoef(na, nb)[0,1]:.3f}$, "
                  f"$n={len(both)}$)", fontsize=9)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}  (n={len(dt)} matched pairs, {len(both)} clips)")


if __name__ == "__main__":
    main()
