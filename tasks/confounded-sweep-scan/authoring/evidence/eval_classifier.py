#!/usr/bin/env python3
"""How accurately can the four classes be told apart at all?

Single-feature thresholds looked clean on eight replicates and stopped being
clean on thirty-two: Tajima's D, which appeared to separate soft sweeps with a
gap of 0.46, overlaps by 1.59 once there are seven soft replicates instead of
four. So the question is not which threshold to pick but how much accuracy is
available, and from which combination of features.

Reports leave-one-out accuracy, which is the honest number: fitting a rule on
all the data and scoring it on the same data measures nothing.
"""
import glob, itertools, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "provenance"))
import config as C

FEATS = ["pi", "tajima_d", "h12", "h2_h1", "nsl_frac"]
BG = {"pi": "bg_pi", "tajima_d": "bg_tajd", "h12": "bg_h12",
      "h2_h1": "bg_h2h1", "nsl_frac": "bg_nsl"}
RATIO = {"pi", "h12", "h2_h1"}
AT = 60_000
CLASSES = ["hard_complete", "hard_partial", "soft_complete", "soft_partial"]


def build():
    X, y = [], []
    for f in sorted(glob.glob(os.path.join(HERE, "training", "*.npz"))):
        d = np.load(f, allow_pickle=True)
        mid = d["mid"]
        sel = np.abs(mid - float(d["sweep_pos"])) <= AT
        row = []
        ok = True
        for k in FEATS:
            v = d[k].astype(float)
            m = sel & np.isfinite(v)
            if not m.any():
                ok = False; break
            x = float(np.median(v[m])); bg = float(d[BG[k]])
            row.append(x / bg if (k in RATIO and bg) else x - bg)
        if ok:
            X.append(row); y.append(str(d["realized_class"]))
    return np.array(X, float), np.array(y)


def loo_centroid(X, y, cols):
    """Leave-one-out nearest-centroid on standardized features."""
    Xs = X[:, cols]
    mu, sd = Xs.mean(0), Xs.sd if False else Xs.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (Xs - mu) / sd
    correct = 0
    conf = {a: {b: 0 for b in CLASSES} for a in CLASSES}
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        cents = {}
        for c in CLASSES:
            m = tr & (y == c)
            if m.sum():
                cents[c] = Z[m].mean(0)
        if not cents:
            continue
        pred = min(cents, key=lambda c: np.linalg.norm(Z[i] - cents[c]))
        conf[y[i]][pred] += 1
        correct += (pred == y[i])
    return correct / len(y), conf


def main():
    X, y = build()
    print(f"{len(y)} replicates: " + ", ".join(
        f"{c}={int((y == c).sum())}" for c in CLASSES))
    print()
    print("leave-one-out accuracy of a nearest-centroid rule, by feature set:")
    best = None
    for r in range(1, len(FEATS) + 1):
        for cols in itertools.combinations(range(len(FEATS)), r):
            acc, _ = loo_centroid(X, y, list(cols))
            if best is None or acc > best[0]:
                best = (acc, cols)
    for r in range(1, len(FEATS) + 1):
        rows = []
        for cols in itertools.combinations(range(len(FEATS)), r):
            acc, _ = loo_centroid(X, y, list(cols))
            rows.append((acc, cols))
        rows.sort(reverse=True)
        acc, cols = rows[0]
        print(f"  best of {r} feature(s): {acc:5.1%}  "
              f"{[FEATS[c] for c in cols]}")
    acc, conf = loo_centroid(X, y, list(best[1]))
    print(f"\noverall best: {acc:.1%} using {[FEATS[c] for c in best[1]]}")
    print("\nconfusion (rows true, columns predicted):")
    hdr = "".join(f"{c[:12]:>14}" for c in CLASSES)
    print(f"{'':<22}{hdr}")
    for a in CLASSES:
        print(f"{a:<22}" + "".join(f"{conf[a][b]:>14}" for b in CLASSES))
    print("\ncentroids on standardized features:")
    Xs = X[:, list(best[1])]
    mu, sd = Xs.mean(0), np.where(Xs.std(0) > 0, Xs.std(0), 1.0)
    Z = (Xs - mu) / sd
    print(f"{'class':<22}" + "".join(f"{FEATS[c][:12]:>14}" for c in best[1]))
    for c in CLASSES:
        m = y == c
        if m.sum():
            print(f"{c:<22}" + "".join(f"{v:>14.2f}" for v in Z[m].mean(0)))
    print("\nraw means and standard deviations (unstandardized):")
    print(f"{'class':<22}" + "".join(f"{FEATS[c][:12]:>16}" for c in best[1]))
    for c in CLASSES:
        m = y == c
        if m.sum():
            print(f"{c:<22}" + "".join(
                f"{X[m][:, c2].mean():>8.3f}+-{X[m][:, c2].std():<7.3f}"
                for c2 in best[1]))


if __name__ == "__main__":
    main()
