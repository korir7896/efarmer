#!/usr/bin/env python3
"""Every number README.md and task.toml quote about the instance, in one place.

The last review's main finding was that the prose had drifted from the code:
feature sets that were no longer used, accuracies measured on a calibration
set that had since grown, a pointer to a table that was not there. The cause
was that each number was retyped by hand into two documents.

This prints them from the sealed truth, the genome config and grading.GATES,
so the documents can cite one source and a reviewer can regenerate it. It
reads the sealed truth, so it is an AUTHORING tool and nothing in the
shipped bundle depends on it.
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(TASK, "tests"))
sys.path.insert(0, os.path.join(HERE, "..", "provenance"))

import config as C
from grading import GATES, TIMED_CLASSES

CLASSES = ("hard_complete", "soft_complete", "hard_partial", "soft_partial")


def main() -> int:
    truth = json.load(open(os.path.join(TASK, "tests", "truth.json")))
    cfg = C.genome_config()
    g = truth["genome"]
    sweeps = sorted(truth["sweeps"], key=lambda s: (s["chrom"], s["position_local"]))

    print("=== genome ===")
    print(f"  {g['n_chromosomes']} chromosomes of {g['chromosome_length_bp']:,} bp "
          f"= {g['n_chromosomes'] * g['chromosome_length_bp'] / 1e6:.0f} Mb")
    print(f"  {C.N_SAMPLE_DIPLOIDS} diploids = {2 * C.N_SAMPLE_DIPLOIDS} haplotypes")
    print(f"  base recombination rate {g['base_recombination_rate']:.1e}, "
          f"mutation rate {g['mutation_rate']:.1e}")
    d = truth["demography"]
    print(f"  history: N_anc {d['N_ancestral']:,} -> N_bot "
          f"{d['N_bottleneck']:,} for {d['bottleneck_duration_gens']} gens "
          f"from {d['T_bottleneck_start_gens_ago']:,} to "
          f"{d['T_bottleneck_end_gens_ago']:,} -> N_cur {d['N_current']:,}")
    print(f"  duration / N_bottleneck = "
          f"{d['bottleneck_duration_gens'] / d['N_bottleneck']:.3f}")

    print("\n=== cold spots ===")
    for s in sorted(cfg.cold_spots, key=lambda x: (x.chrom, x.start)):
        kind = "decoy, no sweep" if s.decoy else "holds a sweep"
        print(f"  chr{s.chrom + 1} {s.start:>10,}-{s.end:<10,} "
              f"{(s.end - s.start) / 1e6:>4.2f} Mb at {s.factor:>5.2f}x  {kind}")
    per_chrom = Counter(s.chrom for s in cfg.cold_spots)
    none_on = [c + 1 for c in range(cfg.n_chrom) if c not in per_chrom]
    print(f"  {len(cfg.cold_spots)} spots on {len(per_chrom)} of "
          f"{cfg.n_chrom} chromosomes; none on chr{none_on}")

    print("\n=== the zoo, as realized ===")
    print(f"  {'id':<6}{'chrom':<7}{'position':>12}{'class':<15}{'s':>8}"
          f"{'done':>7}{'freq':>7}{'orig':>6}  rate")
    for s in sweeps:
        rate = cfg.rate_at(cfg.chrom_offset(s["chrom"] - 1) + s["position_local"])
        cold = "" if rate >= cfg.base_rec_rate * 0.9 else "  COLD"
        done = s.get("completion_gens_ago")
        print(f"  {s['sweep_id']:<6}chr{s['chrom']:<4}{s['position_local']:>12,}"
              f"{s['class']:<15}{s['selection_coefficient']:>8.4f}"
              f"{(round(done) if done else '-'):>7}"
              f"{s.get('final_frequency', 0):>7.3f}"
              f"{s.get('n_origins_surviving', 0):>6}{cold}")

    census = Counter(s["class"] for s in sweeps)
    print(f"\n  census: " + ", ".join(f"{k} {census.get(k, 0)}" for k in CLASSES)
          + f"  (total {len(sweeps)})")
    ss = sorted(s["selection_coefficient"] for s in sweeps)
    print(f"  s from {ss[0]:.4f} to {ss[-1]:.4f}, a factor of "
          f"{ss[-1] / ss[0]:.2f}")
    dated = [s for s in sweeps if s["class"] in TIMED_CLASSES
             and s.get("completion_gens_ago") is not None]
    ages = sorted(s["completion_gens_ago"] for s in dated)
    if ages:
        print(f"  {len(dated)} dateable: ages "
              f"{', '.join(str(round(a)) for a in ages)} "
              f"(a factor of {ages[-1] / ages[0]:.1f})")

    # Is s monotone in age? The zoo is built so it is not.
    hc = [(s["completion_gens_ago"], s["selection_coefficient"])
          for s in dated]
    hc.sort()
    order = [s for _, s in hc]
    print(f"  s by ascending age: {[round(x, 4) for x in order]} -> "
          f"{'monotone (BAD)' if order == sorted(order) or order == sorted(order, reverse=True) else 'non-monotone, as designed'}")

    print("\n=== close pairs ===")
    for a, b in C.CLOSE_PAIRS:
        pa = next(s for s in sweeps if s["sweep_id"] == a)
        pb = next(s for s in sweeps if s["sweep_id"] == b)
        gap = abs(pa["position_local"] - pb["position_local"])
        print(f"  {a} ({pa['class']}) and {b} ({pb['class']}) on "
              f"chr{pa['chrom']}: {gap:,} bp apart, against a dedupe radius "
              f"of {GATES['dedupe_bp']:,}")

    print("\n=== sweeps inside a cold spot ===")
    for sid in C.SWEEPS_IN_COLDSPOT:
        s = next(x for x in sweeps if x["sweep_id"] == sid)
        rate = cfg.rate_at(cfg.chrom_offset(s["chrom"] - 1) + s["position_local"])
        print(f"  {sid} ({s['class']}, s={s['selection_coefficient']:.4f}) at "
              f"rate {rate:.2e}, {cfg.base_rec_rate / rate:.1f}x below base: "
              f"its footprint is that much wider in bp at the same s")

    print("\n=== gates ===")
    for k in ("match_window_bp", "dedupe_bp", "N_ancestral_factor",
              "N_current_factor", "bottleneck_strength_factor",
              "max_false_positives", "min_class_accuracy", "s_within_factor",
              "min_s_accuracy", "min_s_spread", "min_s_rank_corr",
              "min_matched_for_rank", "time_rel_tol", "time_abs_tol_gens",
              "min_time_correct"):
        print(f"  {k:<28}{GATES[k]}")
    rc = GATES["min_recall_per_class"]
    print(f"  {'min_recall_per_class':<28}" + ", ".join(
        f"{k} {rc[k]}/{census.get(k, 0)}" for k in CLASSES))
    print(f"  minimum matched calls to clear recall: {sum(rc.values())} "
          f"of {len(sweeps)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
