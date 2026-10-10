#!/usr/bin/env python3
"""Would the estimators pass their gates? Measured per class on training data.

Reports the fraction of independent single-sweep replicates whose estimated
selection coefficient falls within a factor of 3 of truth, and whose
estimated completion time falls inside the verifier's tolerance. These
fractions are what the pass thresholds have to be set below.
"""
import glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "provenance"))
import config as C

CLASSES = ["hard_recent_complete", "hard_old_complete", "soft_standing", "partial"]
S_FACTOR = 3.0
TIME_REL, TIME_ABS = 0.50, 150


def smooth(v, k=15):
    out = np.full(v.size, np.nan); h = k // 2
    for i in range(v.size):
        a, b = max(0, i - h), min(v.size, i + h + 1)
        seg = v[a:b][np.isfinite(v[a:b])]
        if seg.size:
            out[i] = np.median(seg)
    return out


def trough_width(mid, v, bg, centre, k=15, frac=0.5):
    vs = smooth(v, k)
    near = (np.abs(mid - centre) <= 1_200_000) & np.isfinite(vs)
    if not near.any():
        return 0.0
    vmin = float(vs[near].min()); level = vmin + frac * (bg - vmin)
    order = np.argsort(mid); m, vv = mid[order], vs[order]
    below = np.isfinite(vv) & (vv < level)
    if not below.any():
        return 0.0
    i0 = int(np.argmin(np.abs(m - centre)))
    if not below[i0]:
        cand = np.where(below)[0]
        i0 = int(cand[np.argmin(np.abs(m[cand] - centre))])
        if abs(m[i0] - centre) > 600_000:
            return 0.0
    lo = i0
    while lo - 1 >= 0 and below[lo - 1]:
        lo -= 1
    hi = i0
    while hi + 1 < len(m) and below[hi + 1]:
        hi += 1
    return float(m[hi] - m[lo])


WIDTH_K = {"hard_recent_complete": 0.1371, "hard_old_complete": 0.1143,
           "soft_standing": 0.0750, "partial": 0.0889}
TIME_BIAS = {"hard_recent_complete": 1.617, "hard_old_complete": 1.091}


def main():
    per = {c: {"s_ok": 0, "s_n": 0, "t_ok": 0, "t_n": 0, "s_est": [], "t_est": []}
           for c in CLASSES}
    for f in sorted(glob.glob(os.path.join(HERE, "training", "*.npz"))):
        d = np.load(f, allow_pickle=True)
        cls = str(d["realized_class"])
        if cls not in per:
            continue
        mid = d["mid"]; v = d["pi"].astype(float); bg = float(d["bg_pi"])
        centre = float(d["sweep_pos"]); s_true = float(d["s_true"])

        w = trough_width(mid, v, bg, centre)
        if w > 0:
            s_hat = w * C.BASE_REC_RATE / WIDTH_K[cls]
            per[cls]["s_n"] += 1
            per[cls]["s_est"].append(s_hat / s_true)
            if s_true / S_FACTOR <= s_hat <= s_true * S_FACTOR:
                per[cls]["s_ok"] += 1

        if cls in TIME_BIAS and "completed" in d.files and bool(d["completed"].item()):
            t_true = float(d["completion_gens_ago"])
            near = (np.abs(mid - centre) <= 1_000_000) & np.isfinite(v)
            if near.any() and bg:
                t_hat = (2.0 * C.N_CURRENT * float(v[near].min()) / bg) / TIME_BIAS[cls]
                per[cls]["t_n"] += 1
                per[cls]["t_est"].append(t_hat / t_true)
                if abs(t_hat - t_true) <= max(TIME_ABS, TIME_REL * t_true):
                    per[cls]["t_ok"] += 1

    print("selection coefficient, within a factor of 3 of truth:")
    print(f"  {'class':<24}{'pass':>10}{'rate':>8}{'median est/true':>18}")
    for c in CLASSES:
        p = per[c]
        if p["s_n"]:
            print(f"  {c:<24}{p['s_ok']}/{p['s_n']:<8}{p['s_ok']/p['s_n']:>7.0%}"
                  f"{np.median(p['s_est']):>18.2f}")
    tot_ok = sum(per[c]["s_ok"] for c in CLASSES)
    tot_n = sum(per[c]["s_n"] for c in CLASSES)
    print(f"  {'ALL':<24}{tot_ok}/{tot_n:<8}{tot_ok/tot_n:>7.0%}")

    print("\ncompletion time, within max(150, 50%) of truth:")
    print(f"  {'class':<24}{'pass':>10}{'rate':>8}{'median est/true':>18}")
    for c in CLASSES:
        p = per[c]
        if p["t_n"]:
            print(f"  {c:<24}{p['t_ok']}/{p['t_n']:<8}{p['t_ok']/p['t_n']:>7.0%}"
                  f"{np.median(p['t_est']):>18.2f}")


if __name__ == "__main__":
    main()
