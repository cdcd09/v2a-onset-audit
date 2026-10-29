"""Plot the averaged +-300 ms ETA waveforms (Sec. 3.2).

Shows what the contrast ratio summarises: the event response holds while the
response away from reference onsets rises.

    python -m figures.plot_eta_curves --curves curves.json --out fig_eta.pdf

`curves.json` maps a label to {"event": [...], "control": [...]}, 61 values each
(10 ms frames, index 30 = the reference onset). Convert the JSON from
`python -m instrument.eta --keep_curves` with `analysis.collect_eta_curves`.
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HALF = 30


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--curves", required=True)
    ap.add_argument("--out", default="fig_eta.pdf")
    a = ap.parse_args()

    data = json.load(open(a.curves))
    t = (np.arange(2 * HALF + 1) - HALF) * 10.0        # ms

    fig, axes = plt.subplots(1, len(data), figsize=(3.1 * len(data), 2.7),
                             sharey=False)
    if len(data) == 1:
        axes = [axes]
    for ax, (label, c) in zip(axes, data.items()):
        ev, ct = np.array(c["event"]), np.array(c["control"])
        ax.plot(t, ev, lw=1.8, color="#1f4e79", label="at reference onsets")
        ax.plot(t, ct, lw=1.4, color="#b04a2a", ls="--", label="away from onsets")
        ax.axvline(0, color="0.6", lw=0.7, zorder=0)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("time relative to reference onset (ms)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("onset-strength (z)", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
