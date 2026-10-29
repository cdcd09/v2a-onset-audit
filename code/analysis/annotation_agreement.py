"""Agreement between two independent annotation passes over the benchmark (Sec. S1.4).

The released annotation was produced by one annotator per clip and reviewed by
two others, which is not independent re-annotation and so yields no agreement
statistic. Two annotators therefore re-annotated the whole clip set in separate
lanes, each working from the video with the released timestamps hidden and
without sight of the other lane. This script scores one lane against the other,
and each lane against the release. Candidate pairs are sorted globally by
absolute time difference with an inclusive tolerance boundary. The model event
scorer instead visits reference onsets in order with a strict boundary.

Input is a long-format CSV with one row per (clip, source):

    dataset,clip_id,source,annotator,usable,done,onset_num,onsets_sec

`source` is `gt_release`, `lane_A` or `lane_B`; `onsets_sec` is a
semicolon-separated list of times in seconds. Clips a lane marks unusable carry
no onsets, and agreement is computed on the clips both lanes accept.

    python -m analysis.annotation_agreement \
        --csv /path/to/downloaded-release/metadata/annotation_lanes.csv \
        --out outputs/annotation_agreement.json

No scoring window is applied: unlike the model scores, this measures the
annotation itself, so every annotated onset participates.
"""
import argparse, csv, json, statistics, sys
from pathlib import Path

TOLERANCES_MS = (50, 100, 300)


def parse_onsets(field):
    return sorted(float(x) for x in field.strip().split(";") if x)


def greedy_match(ref, est, tol):
    """Greedy nearest matching, one-to-one, within `tol` seconds.

    Candidate pairs are visited in order of increasing |dt| and accepted while
    both endpoints are still free. Returns (i, j, est_j - ref_i) triples.
    """
    if not ref or not est:
        return []
    pairs = sorted((abs(r - e), i, j, e - r)
                   for i, r in enumerate(ref)
                   for j, e in enumerate(est)
                   if abs(r - e) <= tol)
    used_r, used_e, out = set(), set(), []
    for _, i, j, signed in pairs:
        if i in used_r or j in used_e:
            continue
        used_r.add(i)
        used_e.add(j)
        out.append((i, j, signed))
    return out


def score(clips, table, ref_src, est_src, tol):
    tp = n_ref = n_est = 0
    deltas = []
    for c in clips:
        ref, est = table[c][ref_src], table[c][est_src]
        n_ref += len(ref)
        n_est += len(est)
        m = greedy_match(ref, est, tol)
        tp += len(m)
        deltas += [d for _, _, d in m]
    p = tp / n_est if n_est else 0.0
    r = tp / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "P": round(p, 4), "R": round(r, 4), "F": round(f, 4),
        "matched": tp, "n_ref": n_ref, "n_est": n_est,
        "median_abs_dt_ms": round(statistics.median(abs(d) for d in deltas) * 1e3, 2) if deltas else None,
        "median_signed_dt_ms": round(statistics.median(deltas) * 1e3, 2) if deltas else None,
    }


def cohen_kappa(a, b):
    """Cohen's kappa for two boolean labellings of the same items."""
    n = len(a)
    both = sum(x and y for x, y in zip(a, b))
    neither = sum((not x) and (not y) for x, y in zip(a, b))
    po = (both + neither) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, (po - pe) / (1 - pe) if pe != 1 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    onsets, usable = {}, {}
    seen = set()
    for row in csv.DictReader(open(a.csv)):
        cid, src = row["clip_id"], row["source"]
        if (cid, src) in seen:
            raise ValueError(f"Duplicate annotation lane: {cid}, {src}")
        seen.add((cid, src))
        onsets.setdefault(cid, {})[src] = parse_onsets(row["onsets_sec"])
        usable.setdefault(cid, {})[src] = row["usable"] == "1"

    needed = {"gt_release", "lane_A", "lane_B"}
    complete = sorted(c for c in onsets if needed <= set(onsets[c]))
    incomplete = sorted(set(onsets) - set(complete))

    ua = [usable[c]["lane_A"] for c in complete]
    ub = [usable[c]["lane_B"] for c in complete]
    po, kappa = cohen_kappa(ua, ub)
    both = [c for c in complete if usable[c]["lane_A"] and usable[c]["lane_B"]]
    neither = [c for c in complete if not usable[c]["lane_A"] and not usable[c]["lane_B"]]
    split = [c for c in complete if usable[c]["lane_A"] != usable[c]["lane_B"]]

    out = {
        "n_clips_complete_pair": len(complete),
        "n_clips_incomplete": len(incomplete),
        "clips_incomplete": incomplete,
        "usability": {
            "both": len(both), "neither": len(neither), "split": len(split),
            "observed_agreement": round(po, 4), "cohen_kappa": round(kappa, 3),
            "released_onsets_on_neither": sum(len(onsets[c]["gt_release"]) for c in neither),
            "released_onsets_on_split": sum(len(onsets[c]["gt_release"]) for c in split),
        },
        "n_onsets_on_both_usable": {
            src: sum(len(onsets[c][src]) for c in both)
            for src in ("gt_release", "lane_A", "lane_B")},
        "agreement": {},
    }
    for ref, est, name in (("lane_A", "lane_B", "A_vs_B"),
                           ("gt_release", "lane_A", "release_vs_A"),
                           ("gt_release", "lane_B", "release_vs_B")):
        for tol_ms in TOLERANCES_MS:
            out["agreement"][f"{name}@{tol_ms}"] = score(
                both, onsets, ref, est, tol_ms / 1e3)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))

    u = out["usability"]
    print(f"clips with a complete pair: {out['n_clips_complete_pair']}"
          + (f" (incomplete: {', '.join(incomplete)})" if incomplete else ""))
    print(f"usability: both {u['both']} · neither {u['neither']} · split {u['split']} "
          f"· agreement {u['observed_agreement']} · kappa {u['cohen_kappa']}")
    print(f"released onsets on clips a lane rejected: "
          f"{u['released_onsets_on_neither']} (both) + {u['released_onsets_on_split']} (split)")
    n = out["n_onsets_on_both_usable"]
    print(f"onsets on the {u['both']} unanimous clips: "
          f"A {n['lane_A']} · B {n['lane_B']} · release {n['gt_release']}")
    print(f"{'pair':>14} {'tol':>5} {'P':>7} {'R':>7} {'F':>7} {'med|dt|':>8}")
    for k, v in out["agreement"].items():
        name, tol = k.split("@")
        print(f"{name:>14} {tol:>5} {v['P']:7.4f} {v['R']:7.4f} {v['F']:7.4f} "
              f"{v['median_abs_dt_ms']:8.1f}")


if __name__ == "__main__":
    main()
