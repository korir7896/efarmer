#!/usr/bin/env python3
"""What the obvious approaches achieve on the shipped replicate.

Each baseline differs from the reference in exactly one respect, so the
comparison isolates which piece of the work each gate is actually demanding.
Every baseline is handed the reference's own classification and estimation
step, so differences here are differences in DETECTION and CALIBRATION, not
in downstream bookkeeping.

  single_statistic_*   one statistic, cut at the genome's own 99.9th
                       percentile. No demographic inference, no null
                       simulation, no awareness of local recombination rate.
                       This is a scan run at its defaults.
  multi_constant_null  all four detection statistics, but the null is
                       simulated under a constant-size population whose size
                       is read off mean diversity. Tests whether inferring
                       the bottleneck matters.
  multi_rate_blind     correct history, correct null simulated on the real
                       map, but one pooled threshold instead of comparing
                       each window against null windows of similar local
                       rate. Tests whether handling the cold spots matters.
  reference            the full pipeline.
"""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(TASK, "solution"))
sys.path.insert(0, os.path.join(TASK, "tests"))
sys.path.insert(0, os.path.join(HERE, "..", "provenance"))

from pipeline import characterize as CH
from pipeline import demography as DEM
from pipeline import detect as DET
from pipeline import sweepstats as SS
import grading as G

WIN, STEP = 50_000, 10_000
MU = 2.0e-8


def naive_constant_size(obs_pi, mu):
    """N from mean diversity alone: pi = 4*N*mu."""
    return obs_pi / (4.0 * mu)


def characterise(obs, regions, med_rate, n_current):
    bins = DET.rate_bin(obs["rate"], med_rate)
    bg = CH.background_levels(obs, regions, bins)
    calls = []
    for r in regions:
        b = int(bins[r["peak_index"]])
        lev = bg.get(b) or list(bg.values())[0]
        f = CH.features(obs, r, lev)
        klass = CH.classify(f)
        s_hat = CH.estimate_s(obs, r, lev["pi"],
                              float(obs["rate"][r["peak_index"]]), klass)
        t_hat = CH.estimate_time(f, klass, n_current)
        calls.append({"chrom": int(r["chrom"]), "position": int(r["position"]),
                      "class": klass,
                      "selection_coefficient": float(s_hat) if np.isfinite(s_hat) else 0.01,
                      "time_gens_ago": None if t_hat is None else float(round(t_hat))})
    return calls


def single_statistic_regions(obs, stat, direction, pct=99.9):
    v = obs[stat].astype(float)
    ok = np.isfinite(v)
    if direction == "low":
        thr = np.percentile(v[ok], 100 - pct)
        score = -(v - np.nanmean(v[ok])) / (np.nanstd(v[ok]) or 1.0)
        hot_thr = -(thr - np.nanmean(v[ok])) / (np.nanstd(v[ok]) or 1.0)
    else:
        thr = np.percentile(v[ok], pct)
        score = (v - np.nanmean(v[ok])) / (np.nanstd(v[ok]) or 1.0)
        hot_thr = (thr - np.nanmean(v[ok])) / (np.nanstd(v[ok]) or 1.0)
    return DET.call_regions(obs, score, hot_thr)


def pooled_threshold_regions(obs, null, med_rate, expected_false=0.3, n_reps=10):
    """Rate-blind: one null distribution for the whole genome."""
    score = np.full(obs["pi"].size, -np.inf)
    for stat, direction in DET.DETECT_STATS:
        o = obs[stat].astype(float)
        nv = null[stat].astype(float); nv = nv[np.isfinite(nv)]
        mu_, sd = float(np.mean(nv)), float(np.std(nv))
        if sd <= 0:
            continue
        z = (o - mu_) / sd
        score = np.maximum(score, -z if direction == "low" else z)
    # same false-positive budget, estimated from the pooled null
    nscore = np.full(null["pi"].size, -np.inf)
    for stat, direction in DET.DETECT_STATS:
        o = null[stat].astype(float)
        nv = o[np.isfinite(o)]
        mu_, sd = float(np.mean(nv)), float(np.std(nv))
        if sd <= 0:
            continue
        z = (o - mu_) / sd
        nscore = np.maximum(nscore, -z if direction == "low" else z)
    fin = nscore[np.isfinite(nscore)]
    per_genome = fin.size / max(1, n_reps)
    q = min(max(1.0 - expected_false / per_genome, 0.5), 1 - 1e-9)
    return DET.call_regions(obs, score, float(np.quantile(fin, q)))


def grade(truth, submission):
    """Which gates does this submission clear?"""
    try:
        dem, calls = G.parse_submission(submission, truth["genome"]["n_chromosomes"],
                                        truth["genome"]["chromosome_length_bp"])
    except G.SchemaError as e:
        return {"schema": f"invalid: {e}"}, False
    calls = G.dedupe(calls, G.GATES["dedupe_bp"])
    pairs, un_t, un_c = G.match(truth["sweeps"], calls, G.GATES["match_window_bp"])

    res = {}
    td = truth["demography"]
    for key, gkey in (("N_ancestral", "N_ancestral_factor"),
                      ("N_current", "N_current_factor")):
        f = G.GATES[gkey]; t = float(td[key]); g = float(dem[key])
        res[f"demog_{key}"] = (t / f <= g <= t * f)
    true_str = ((float(td["T_bottleneck_start_gens_ago"]) - float(td["T_bottleneck_end_gens_ago"]))
                / float(td["N_bottleneck"]))
    dur = float(dem["T_bottleneck_start_gens_ago"]) - float(dem["T_bottleneck_end_gens_ago"])
    got_str = dur / float(dem["N_bottleneck"]) if dur > 0 else -1
    fs = G.GATES["bottleneck_strength_factor"]
    res["demog_bottleneck_strength"] = (true_str / fs <= got_str <= true_str * fs)
    tf = G.GATES["T_bottleneck_end_frac"]
    res["demog_T_end"] = abs(float(dem["T_bottleneck_end_gens_ago"])
                             - float(td["T_bottleneck_end_gens_ago"])) <= tf * float(td["T_bottleneck_end_gens_ago"])
    res["demog_is_bottleneck"] = (float(dem["N_bottleneck"]) < min(float(dem["N_ancestral"]),
                                                                   float(dem["N_current"])))

    found = {}
    for t, c, d in pairs:
        found[t["class"]] = found.get(t["class"], 0) + 1
    recall_ok = True
    recall_detail = {}
    for klass, need in G.GATES["min_recall_per_class"].items():
        tot = sum(1 for t in truth["sweeps"] if t["class"] == klass)
        got = found.get(klass, 0)
        recall_detail[klass] = f"{got}/{tot}(need {need})"
        if got < need:
            recall_ok = False
    res["recall"] = recall_ok
    res["false_positives"] = len(un_c) <= G.GATES["max_false_positives"]

    if pairs:
        right = sum(1 for t, c, _ in pairs if c["class"] == t["class"])
        res["class_accuracy"] = (right / len(pairs)) >= G.GATES["min_class_accuracy"]
        fsf = G.GATES["s_within_factor"]
        ok_s = sum(1 for t, c, _ in pairs
                   if float(t["selection_coefficient"]) / fsf <= c["s"] <= float(t["selection_coefficient"]) * fsf)
        res["s_accuracy"] = (ok_s / len(pairs)) >= G.GATES["min_s_accuracy"]
        by_class = {}
        for t, c, _ in pairs:
            if t["class"] in G.TIMED_CLASSES and t["completion_gens_ago"] is not None:
                by_class.setdefault(t["class"], []).append((t, c))
        tok = bool(by_class)
        for klass, items in by_class.items():
            good = 0
            for t, c in items:
                tt = float(t["completion_gens_ago"])
                tol = max(G.GATES["time_abs_tol_gens"], G.GATES["time_rel_tol"] * tt)
                if c["time"] is not None and abs(c["time"] - tt) <= tol:
                    good += 1
            if good < max(1, int(round(G.GATES["min_time_accuracy_per_class"] * len(items)))):
                tok = False
        res["timing"] = tok
    else:
        res["class_accuracy"] = res["s_accuracy"] = res["timing"] = False

    res["_n_calls"] = len(calls)
    res["_n_false_positives"] = len(un_c)
    res["_recall_detail"] = recall_detail
    passed = all(v for k, v in res.items() if not k.startswith("_"))
    return res, passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory with variants.vcf.gz and recombination_map.bed")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--n-null", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "baselines_result.json"))
    a = ap.parse_args()

    truth = json.load(open(a.truth))
    chroms = SS.load_vcf(os.path.join(a.data, "variants.vcf.gz"))
    rm = DET.load_rate_map(os.path.join(a.data, "recombination_map.bed"))
    med_rate = rm.median_rate()
    n_dip = next(iter(chroms.values())).n_hap // 2

    print("scanning observed genome", flush=True)
    obs = DET.scan_genome(chroms, rm.chrom_len, rm, WIN, STEP)
    obs_sum = DEM.observed_summaries(chroms, rm.chrom_len)

    print("fitting true-shape history (shared by the calibrated baselines)", flush=True)
    hist, _ = DEM.fit_demography(obs_sum, mu=MU, rate=med_rate, n_dip=n_dip,
                                 seed=20_240_101, log=lambda m: None)
    print(f"  {hist.as_dict()}", flush=True)

    print(f"simulating {a.n_null} null replicates under the fitted history", flush=True)
    null = DET.build_null(hist, rm, MU, n_dip, WIN, STEP, n_reps=a.n_null,
                          seed=20_240_101 + 5000, log=lambda m: None)

    n_naive = naive_constant_size(obs_sum["pi"], MU)
    naive_demog = {"N_ancestral": n_naive, "N_bottleneck": n_naive,
                   "N_current": n_naive,
                   "T_bottleneck_start_gens_ago": 3000.0,
                   "T_bottleneck_end_gens_ago": 2000.0}
    print(f"  naive constant-size N from diversity alone: {n_naive:.0f}", flush=True)

    strategies = {}
    for stat, direction in (("pi", "low"), ("tajima_d", "low"),
                            ("h12", "high"), ("nsl_frac", "high")):
        regions = single_statistic_regions(obs, stat, direction)
        strategies[f"single_statistic_{stat}"] = (naive_demog, regions)

    print("null under a constant-size population (no bottleneck inferred)", flush=True)
    flat = DEM.History(n_naive, n_naive * 0.999, n_naive, 3000.0, 2000.0)
    null_flat = DET.build_null(flat, rm, MU, n_dip, WIN, STEP,
                               n_reps=max(3, a.n_null // 2),
                               seed=20_240_101 + 9000, log=lambda m: None)
    score_flat = DET.sweep_scores(obs, null_flat, med_rate)
    thr_flat = DET.null_score_threshold(null_flat, med_rate, max(3, a.n_null // 2))
    strategies["multi_constant_null"] = (naive_demog,
                                         DET.call_regions(obs, score_flat, thr_flat))

    strategies["multi_rate_blind"] = (hist.as_dict(),
                                      pooled_threshold_regions(obs, null, med_rate,
                                                               n_reps=a.n_null))

    score = DET.sweep_scores(obs, null, med_rate)
    thr = DET.null_score_threshold(null, med_rate, a.n_null)
    strategies["reference"] = (hist.as_dict(), DET.call_regions(obs, score, thr))

    out = {}
    print(f"\n{'strategy':<28}{'calls':>6}{'FP':>4}  gates failed")
    for name, (dem, regions) in strategies.items():
        calls = characterise(obs, regions, med_rate, float(dem["N_current"]))
        sub = {"demography": dem, "sweeps": calls}
        res, passed = grade(truth, sub)
        failed = [k for k, v in res.items() if not k.startswith("_") and not v]
        out[name] = {"passed": passed, "gates": {k: v for k, v in res.items()},
                     "submission": sub}
        print(f"{name:<28}{res['_n_calls']:>6}{res['_n_false_positives']:>4}  "
              f"{'PASSES ALL GATES' if passed else ', '.join(failed)}")
        print(f"{'':<28}{'':>6}{'':>4}  recall {res['_recall_detail']}")
    json.dump(out, open(a.out, "w"), indent=2, default=str)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
