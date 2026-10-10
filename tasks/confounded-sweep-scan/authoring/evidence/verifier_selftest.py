#!/usr/bin/env python3
"""Run the real verifier against a perfect submission and a battery of cheats.

The point is the cheats: a verifier that a constant answer, a carpet of calls
or a copy of the genome-wide average can satisfy is not grading the work.
Every row marked EXPECT fail must fail.

By default this grades against the SHIPPED truth file, tests/truth.json, so
the cheats are measured against the selection coefficients and completion
times the agent is actually asked to recover. `--mock` substitutes a
stand-in truth built from the generator config with plausible realized
values, which is useful before a replicate has been generated but says
nothing about the shipped one.

Two of the cheats are not fixed submissions but searches: the single
selection coefficient and the single completion time that satisfy the most
instances. Reporting the best a constant can do is the only way to show the
gate is not merely strict but unsatisfiable without doing the work.
"""
import argparse, json, math, os, subprocess, sys, tempfile, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, "..", ".."))
TESTS = os.path.join(TASK, "tests")
sys.path.insert(0, os.path.join(HERE, "..", "provenance"))
sys.path.insert(0, TESTS)

import config as C
import sweeps as S
from grading import GATES, TIMED_CLASSES, time_tolerance


def shipped_truth():
    p = os.path.join(TESTS, "truth.json")
    if not os.path.exists(p):
        raise SystemExit(f"no shipped truth at {p}; pass --mock to use a stand-in")
    return json.load(open(p))


def mock_truth():
    """A stand-in truth with plausible realized values, for logic testing."""
    cfg = C.genome_config()
    plans = S.build_plans(cfg, C.MASTER_SEED)
    # Per-sweep realized values, taken as the midpoint of each sweep's own
    # acceptance window, NOT one value per class: giving every sweep of a
    # class the same completion time would make a constant answer pass the
    # timing gate on the mock while failing on the shipped replicate, which
    # is the opposite of what a self-test is for.
    sweeps = []
    for p in plans:
        c = p.position // cfg.chrom_len
        rule = C.ACCEPT[p.sweep_id]
        if rule["must_complete"]:
            done = (rule["done_min"] + rule["done_max"]) // 2
            freq = 1.0
        else:
            done = None
            freq = (rule["freq_min"] + rule["freq_max"]) / 2.0
        sweeps.append({
            "sweep_id": p.sweep_id, "chrom": c + 1,
            "position_global": p.position,
            "position_local": p.position - cfg.chrom_offset(c),
            "selection_coefficient": p.s, "class": p.intended_class,
            "completion_gens_ago": done, "final_frequency": freq,
            "completed": done is not None,
        })
    return {
        "genome": {"n_chromosomes": cfg.n_chrom,
                   "chromosome_length_bp": cfg.chrom_len},
        "demography": C.demography().to_dict(),
        "sweeps": sweeps,
    }


def run_verifier(truth, submission, tmp):
    tdir = os.path.join(tmp, "t"); os.makedirs(tdir, exist_ok=True)
    tp = os.path.join(tdir, "truth.json")
    json.dump(truth, open(tp, "w"))
    sp = os.path.join(tdir, "results.json")
    if os.path.exists(sp):
        os.remove(sp)          # otherwise a previous case's file is graded
    if submission is not None:
        json.dump(submission, open(sp, "w"))
    env = dict(os.environ, TRUTH_PATH=tp, SUBMISSION_PATH=sp,
               PYTHONPATH=TESTS)
    tf = os.path.join(TESTS, "test_sweep_scan.py")
    if not os.path.exists(tf):
        raise SystemExit(f"cannot find the verifier at {tf}")
    r = subprocess.run([sys.executable, "-m", "pytest", tf,
                        "-p", "no:cacheprovider", "-q", "--no-header"],
                       capture_output=True, text=True, env=env)
    if r.returncode not in (0, 1):
        raise SystemExit(f"pytest itself failed (rc={r.returncode}):\n{r.stdout}\n{r.stderr}")
    tail = [l for l in (r.stdout or "").strip().splitlines() if l.strip()]
    return r.returncode == 0, (tail[-1] if tail else "")


def perfect(truth):
    out = []
    for s in truth["sweeps"]:
        c = {"chrom": s["chrom"], "position": s["position_local"],
             "class": s["class"],
             "selection_coefficient": s["selection_coefficient"]}
        if s["completion_gens_ago"] is not None:
            c["time_gens_ago"] = s["completion_gens_ago"]
        out.append(c)
    return {"demography": truth["demography"], "sweeps": out}


def best_constant_s(truth):
    """The single coefficient satisfying the most magnitude tests."""
    f = GATES["s_within_factor"]
    tru = [float(s["selection_coefficient"]) for s in truth["sweeps"]]
    best, best_n = None, -1
    lo, hi = min(tru) / f, max(tru) * f
    steps = 4000
    for i in range(steps + 1):
        c = lo * (hi / lo) ** (i / steps)
        n = sum(1 for t in tru if t / f <= c <= t * f)
        if n > best_n:
            best, best_n = c, n
    return best, best_n, len(tru)


def best_constant_time(truth):
    """The single completion time satisfying the most timing tests."""
    items = [float(s["completion_gens_ago"]) for s in truth["sweeps"]
             if s["class"] in TIMED_CLASSES
             and s["completion_gens_ago"] is not None]
    best, best_n = None, -1
    for t in range(1, 6001):
        n = sum(1 for tt in items if abs(t - tt) <= time_tolerance(tt))
        if n > best_n:
            best, best_n = float(t), n
    return best, best_n, len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true",
                    help="grade against a stand-in truth instead of the shipped one")
    a = ap.parse_args()

    truth = mock_truth() if a.mock else shipped_truth()
    which = "MOCK truth built from the generator config" if a.mock \
        else os.path.join(TESTS, "truth.json")
    CL = truth["genome"]["chromosome_length_bp"]
    NC = truth["genome"]["n_chromosomes"]
    cases = []

    cases.append(("perfect answer", True, perfect(truth)))

    cases.append(("no file at all", False, None))
    cases.append(("empty object", False, {}))
    cases.append(("no sweeps", False, {"demography": truth["demography"], "sweeps": []}))

    # carpet bombing: a call every 100 kb across the genome
    carpet = {"demography": truth["demography"], "sweeps": []}
    for ch in range(1, NC + 1):
        for p in range(200_000, CL, 100_000):
            carpet["sweeps"].append({"chrom": ch, "position": p,
                                     "class": "hard_complete",
                                     "selection_coefficient": 0.03,
                                     "time_gens_ago": 300})
    cases.append(("carpet every 100kb", False, carpet))

    # right positions but one constant class
    for klass in ("hard_complete", "soft_partial"):
        one_class = perfect(truth)
        for s in one_class["sweeps"]:
            s["class"] = klass
            s.setdefault("time_gens_ago", 300)
        cases.append((f"positions right, all {klass}", False, one_class))

    # --- the s gate, three ways ---
    bs, bs_n, bs_tot = best_constant_s(truth)
    for label, val in (("s=0.3 (far off)", 0.3),
                       ("s at the true geometric mean", None),
                       (f"s at the best possible constant {bs:.4f}", bs)):
        const_s = perfect(truth)
        if val is None:
            tru = [float(s["selection_coefficient"]) for s in truth["sweeps"]]
            import math
            val = math.exp(sum(math.log(t) for t in tru) / len(tru))
        for s in const_s["sweeps"]:
            s["selection_coefficient"] = val
        cases.append((f"constant {label}", False, const_s))

    # a constant with noise on it: passes magnitude and spread, and is caught
    # only by the ordering gate. Reported rather than asserted, because at
    # nine matched calls that gate escapes roughly one time in five and the
    # bundle says so (see GATES in tests/grading.py).
    import random
    rng = random.Random(20251010)
    jit = perfect(truth)
    for s_ in jit["sweeps"]:
        s_["selection_coefficient"] = round(0.0245 * math.exp(
            rng.uniform(-0.9, 0.9)), 5)
    cases.append(("constant s with noise (see note)", False, jit))

    # magnitudes right, ordering destroyed: the rank gate alone must catch it
    shuffled = perfect(truth)
    tru_s = [float(s["selection_coefficient"]) for s in truth["sweeps"]]
    rev = sorted(tru_s, reverse=True)
    rank = {v: i for i, v in enumerate(sorted(tru_s))}
    for s in shuffled["sweeps"]:
        s["selection_coefficient"] = rev[rank[float(s["selection_coefficient"])]]
    cases.append(("s values reversed in rank", False, shuffled))

    # --- the timing gate, three ways ---
    bt, bt_n, bt_tot = best_constant_time(truth)
    for label, val in (("time=300", 300.0), ("time=1600", 1600.0),
                       (f"the best possible constant time {bt:.0f}", bt)):
        const_t = perfect(truth)
        for s in const_t["sweeps"]:
            if "time_gens_ago" in s:
                s["time_gens_ago"] = val
        cases.append((f"constant {label}", False, const_t))

    # one constant per class: what the previous, age-based class scheme let
    # through. With age out of the labels this is just a constant again.
    by_class = {}
    for s in truth["sweeps"]:
        if s["completion_gens_ago"] is not None:
            by_class.setdefault(s["class"], []).append(float(s["completion_gens_ago"]))
    means = {k: sum(v) / len(v) for k, v in by_class.items()}
    cc = perfect(truth)
    for s in cc["sweeps"]:
        if "time_gens_ago" in s:
            s["time_gens_ago"] = means.get(s["class"], 300.0)
    cases.append(("one constant time per class", False, cc))

    # timing scaled by two
    t2 = perfect(truth)
    for s_ in t2["sweeps"]:
        if "time_gens_ago" in s_:
            s_["time_gens_ago"] = s_["time_gens_ago"] * 2
    cases.append(("all completion times doubled", False, t2))

    # demography guessed as a constant-size population
    flat = perfect(truth)
    flat["demography"] = {"N_ancestral": 10000, "N_bottleneck": 10000,
                          "N_current": 10000,
                          "T_bottleneck_start_gens_ago": 3400,
                          "T_bottleneck_end_gens_ago": 3000}
    cases.append(("no bottleneck inferred", False, flat))

    # whole classes dropped
    for klass in sorted({s["class"] for s in truth["sweeps"]}):
        drop = perfect(truth)
        drop["sweeps"] = [s for s in drop["sweeps"] if s["class"] != klass]
        cases.append((f"misses every {klass}", False, drop))

    # off by 400 kb
    shifted = perfect(truth)
    for s in shifted["sweeps"]:
        s["position"] = min(CL, s["position"] + 400_000)
    cases.append(("all positions off by 400kb", False, shifted))

    # NaN injection
    nan_sub = perfect(truth)
    nan_sub["sweeps"][0]["selection_coefficient"] = float("nan")
    cases.append(("NaN selection coefficient", False, nan_sub))

    # a complete class with no time at all: the schema must reject it
    no_time = perfect(truth)
    for s in no_time["sweeps"]:
        s.pop("time_gens_ago", None)
    cases.append(("complete sweeps with no time_gens_ago", False, no_time))

    # all calls parked at the middle of each chromosome
    centre = {"demography": truth["demography"], "sweeps": [
        {"chrom": ch, "position": CL // 2, "class": "hard_complete",
         "selection_coefficient": 0.03, "time_gens_ago": 300}
        for ch in range(1, NC + 1)]}
    cases.append(("all calls at chromosome centres", False, centre))

    # perfect plus one false positive: allowed
    fp1 = perfect(truth)
    fp1["sweeps"].append({"chrom": 1, "position": 3_800_000,
                          "class": "hard_complete",
                          "selection_coefficient": 0.03, "time_gens_ago": 300})
    cases.append(("perfect plus 1 cold-spot call", True, fp1))

    # perfect plus three false positives: not allowed
    fp3 = perfect(truth)
    for ch, p in ((1, 3_800_000), (2, 7_600_000), (3, 5_700_000)):
        fp3["sweeps"].append({"chrom": ch, "position": p,
                              "class": "hard_complete",
                              "selection_coefficient": 0.03,
                              "time_gens_ago": 300})
    cases.append(("perfect plus 3 cold-spot calls", False, fp3))

    tmp = tempfile.mkdtemp()
    try:
        print(f"grading against: {which}")
        print("a constant s is excluded by the spread gate with certainty; "
              "the magnitude test alone does not exclude it.")
        print(f"best constant s = {bs:.4f} satisfies the magnitude test on "
              f"{bs_n}/{bs_tot} sweeps (needs "
              f"{GATES['min_s_accuracy']:.2f} of matched, plus the spread "
              f"and ordering tests)")
        print(f"best constant time = {bt:.0f} satisfies "
              f"{bt_n}/{bt_tot} of the dated sweeps "
              f"(needs {GATES['min_time_correct']})")
        print()
        print(f"{'case':<48}{'expect':>8}{'got':>8}   verdict")
        bad = 0
        for name, expect_pass, sub in cases:
            got, last = run_verifier(truth, sub, tmp)
            verdict = "ok" if got == expect_pass else "*** WRONG ***"
            if got != expect_pass:
                bad += 1
            print(f"{name:<48}{str(expect_pass):>8}{str(got):>8}   {verdict}")
        print()
        print(f"{len(cases)} cases; "
              + ("all verifier self-tests behaved as intended" if bad == 0
                 else f"{bad} verifier self-test(s) misbehaved"))
        return 1 if bad else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
