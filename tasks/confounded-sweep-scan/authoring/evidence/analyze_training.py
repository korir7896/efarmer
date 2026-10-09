#!/usr/bin/env python3
"""Per-class signal, localisation spread, and estimator calibration.

Answers the questions the gates depend on:
  * which statistic detects which class, and how strongly;
  * how far the best window sits from the true selected site, which is what
    the verifier's match window has to accommodate;
  * whether the four classes are separable by their feature profile;
  * whether fixation time and s can be recovered, and with what bias.
"""
import glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "provenance"))
import config as C

FEATS = ["pi", "tajima_d", "h12", "h2_h1", "nsl_frac"]
SEARCH = 1_500_000      # look for the extremum within this of the truth


def load_all():
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, "training", "*.npz"))):
        d = np.load(f, allow_pickle=True)
        lab = str(d["label"])
        out.setdefault(lab, []).append(d)
    return out


def extremum(d, feat, direction):
    mid = d["mid"]; v = d[feat].astype(float)
    pos = float(d["sweep_pos"])
    near = (np.abs(mid - pos) <= SEARCH) & np.isfinite(v)
    if not near.any():
        return None, None
    idx = np.where(near)[0]
    j = idx[np.argmin(v[idx])] if direction == "low" else idx[np.argmax(v[idx])]
    return float(mid[j]), float(v[j])


DIRECTION = {"pi": "low", "tajima_d": "low", "h12": "high",
             "h2_h1": "low", "nsl_frac": "high"}


def main():
    data = load_all()
    if not data:
        print("no training replicates yet"); return

    print("=== signal at the sweep, relative to that replicate's background ===")
    print("(pi and H2/H1 as ratios; Tajima's D and nSL as differences; H12 as a ratio)\n")
    hdr = f"{'class':<22}{'n':>3}" + "".join(f"{f:>13}" for f in FEATS)
    print(hdr)
    for lab, reps in sorted(data.items()):
        cells = []
        for f in FEATS:
            vals = []
            for d in reps:
                _, v = extremum(d, f, DIRECTION[f])
                if v is None or not np.isfinite(v):
                    continue
                bg = float(d[{"pi": "bg_pi", "tajima_d": "bg_tajd", "h12": "bg_h12",
                              "h2_h1": "bg_h2h1", "nsl_frac": "bg_nsl"}[f]])
                if f in ("pi", "h12", "h2_h1"):
                    vals.append(v / bg if bg else np.nan)
                else:
                    vals.append(v - bg)
            cells.append(f"{np.nanmedian(vals):>13.3f}" if vals else f"{'-':>13}")
        print(f"{lab:<22}{len(reps):>3}" + "".join(cells))

    print("\n=== localisation error of the best window, |argmin - true| in kb ===")
    print(f"{'class':<22}" + "".join(f"{f:>13}" for f in FEATS))
    allerr = {f: [] for f in FEATS}
    for lab, reps in sorted(data.items()):
        cells = []
        for f in FEATS:
            errs = []
            for d in reps:
                m, v = extremum(d, f, DIRECTION[f])
                if m is None:
                    continue
                errs.append(abs(m - float(d["sweep_pos"])) / 1000.0)
            if errs:
                allerr[f] += errs
                cells.append(f"{np.median(errs):>8.0f}/{np.max(errs):<4.0f}")
            else:
                cells.append(f"{'-':>13}")
        print(f"{lab:<22}" + "".join(cells))
    print("\n(median/max per class, kb)")
    print("\npooled across classes:")
    for f in FEATS:
        e = np.array(allerr[f])
        if e.size:
            print(f"  {f:<10} median {np.median(e):>6.0f} kb   p90 {np.percentile(e,90):>6.0f} kb"
                  f"   max {e.max():>6.0f} kb")

    print("\n=== completion time from diversity recovery:  t_hat = 2*N_cur*pi_min/pi_bg ===")
    print(f"{'class':<22}{'true':>8}{'t_hat':>9}{'ratio':>8}")
    for lab, reps in sorted(data.items()):
        rows = []
        for d in reps:
            if not bool(d["completed"].item()):
                continue
            _, pimin = extremum(d, "pi", "low")
            bg = float(d["bg_pi"])
            if pimin is None or not bg:
                continue
            t_hat = 2.0 * C.N_CURRENT * (pimin / bg)
            rows.append((float(d["completion_gens_ago"]), t_hat))
        if rows:
            tr = np.array([r[0] for r in rows]); th = np.array([r[1] for r in rows])
            print(f"{lab:<22}{np.median(tr):>8.0f}{np.median(th):>9.0f}"
                  f"{np.median(th/tr):>8.2f}")

    print("\n=== width of the diversity trough (pi < 0.5*background), kb ===")
    print(f"{'class':<22}{'s_true':>8}{'width':>9}{'w*r':>10}")
    for lab, reps in sorted(data.items()):
        ws, ss = [], []
        for d in reps:
            mid = d["mid"]; v = d["pi"].astype(float); bg = float(d["bg_pi"])
            pos = float(d["sweep_pos"])
            near = (np.abs(mid - pos) <= SEARCH) & np.isfinite(v)
            if not near.any(): continue
            below = near & (v < 0.5 * bg)
            if not below.any(): continue
            ws.append((mid[below].max() - mid[below].min()) / 1000.0)
            ss.append(float(d["s_true"]))
        if ws:
            w = np.median(ws)
            print(f"{lab:<22}{np.median(ss):>8.3f}{w:>9.0f}"
                  f"{w*1000*C.BASE_REC_RATE:>10.5f}")


if __name__ == "__main__":
    main()
