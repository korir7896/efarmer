#!/usr/bin/env python3
"""Run the verifier against a perfect submission and a battery of cheap ones.

The point is the cheap ones: a verifier that a constant answer, a carpet of
calls, or a copy of the genome-wide average can satisfy is not grading the
work.  Every row marked EXPECT fail must fail.
"""
import json, os, subprocess, sys, tempfile, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.abspath(os.path.join(HERE, "..", "..", "tests"))
sys.path.insert(0, os.path.join(HERE, "..", "provenance"))

import config as C
import sweeps as S


def mock_truth():
    """A stand-in truth with plausible realized values, for logic testing."""
    cfg = C.genome_config()
    plans = S.build_plans(cfg, C.MASTER_SEED)
    realized = {
        "hard_recent_complete": dict(fix=300, freq=1.0),
        "hard_old_complete": dict(fix=1600, freq=1.0),
        "soft_standing": dict(fix=500, freq=1.0),
        "partial": dict(fix=None, freq=0.68),
    }
    sweeps = []
    for p in plans:
        c = p.position // cfg.chrom_len
        r = realized[p.intended_class]
        sweeps.append({
            "sweep_id": p.sweep_id, "chrom": c + 1,
            "position_global": p.position,
            "position_local": p.position - cfg.chrom_offset(c),
            "selection_coefficient": p.s, "class": p.intended_class,
            "completion_gens_ago": r["fix"], "final_frequency": r["freq"],
            "completed": r["fix"] is not None,
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
               PYTHONPATH=os.path.abspath(TESTS))
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
    return {
        "demography": truth["demography"],
        "sweeps": [{"chrom": s["chrom"], "position": s["position_local"],
                    "class": s["class"],
                    "selection_coefficient": s["selection_coefficient"],
                    "time_gens_ago": s["completion_gens_ago"]}
                   for s in truth["sweeps"]],
    }


def main():
    truth = mock_truth()
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
                                     "class": "hard_recent_complete",
                                     "selection_coefficient": 0.03,
                                     "time_gens_ago": 300})
    cases.append(("carpet every 100kb", False, carpet))

    # right positions but one constant class
    one_class = perfect(truth)
    for s in one_class["sweeps"]:
        s["class"] = "hard_recent_complete"
    cases.append(("positions right, all one class", False, one_class))

    # right positions but constant s far from truth
    const_s = perfect(truth)
    for s in const_s["sweeps"]:
        s["selection_coefficient"] = 0.3
    cases.append(("positions right, constant s=0.3", False, const_s))

    # right positions but constant timing
    const_t = perfect(truth)
    for s in const_t["sweeps"]:
        s["time_gens_ago"] = 300
    cases.append(("positions right, constant time=300", False, const_t))

    # demography guessed as a constant-size population
    flat = perfect(truth)
    flat["demography"] = {"N_ancestral": 10000, "N_bottleneck": 10000,
                          "N_current": 10000,
                          "T_bottleneck_start_gens_ago": 3400,
                          "T_bottleneck_end_gens_ago": 3000}
    cases.append(("no bottleneck inferred", False, flat))

    # only the three easy recent sweeps
    easy = perfect(truth)
    easy["sweeps"] = [s for s in easy["sweeps"] if s["class"] == "hard_recent_complete"]
    cases.append(("only recent hard sweeps", False, easy))

    # everything but the old ones
    no_old = perfect(truth)
    no_old["sweeps"] = [s for s in no_old["sweeps"] if s["class"] != "hard_old_complete"]
    cases.append(("misses both old sweeps", False, no_old))

    # off by 400 kb
    shifted = perfect(truth)
    for s in shifted["sweeps"]:
        s["position"] = min(CL, s["position"] + 400_000)
    cases.append(("all positions off by 400kb", False, shifted))

    # NaN injection
    nan_sub = perfect(truth)
    nan_sub["sweeps"][0]["selection_coefficient"] = float("nan")
    cases.append(("NaN selection coefficient", False, nan_sub))

    # all calls parked at the middle of each chromosome
    centre = {"demography": truth["demography"], "sweeps": [
        {"chrom": ch, "position": CL // 2, "class": "hard_recent_complete",
         "selection_coefficient": 0.03, "time_gens_ago": 300}
        for ch in range(1, NC + 1)]}
    cases.append(("all calls at chromosome centres", False, centre))

    # timing scaled by two
    t2 = perfect(truth)
    for s_ in t2["sweeps"]:
        if s_["time_gens_ago"] is not None:
            s_["time_gens_ago"] = s_["time_gens_ago"] * 2
    cases.append(("all fixation times doubled", False, t2))

    # a constant that favours the old sweeps instead
    t_old = perfect(truth)
    for s_ in t_old["sweeps"]:
        s_["time_gens_ago"] = 1600
    cases.append(("positions right, constant time=1600", False, t_old))

    # perfect plus one false positive: allowed
    fp1 = perfect(truth)
    fp1["sweeps"].append({"chrom": 1, "position": 3_800_000,
                          "class": "hard_recent_complete",
                          "selection_coefficient": 0.03, "time_gens_ago": 300})
    cases.append(("perfect plus 1 cold-spot call", True, fp1))

    # perfect plus three false positives: not allowed
    fp3 = perfect(truth)
    for ch, p in ((1, 3_800_000), (2, 7_600_000), (3, 5_700_000)):
        fp3["sweeps"].append({"chrom": ch, "position": p,
                              "class": "hard_recent_complete",
                              "selection_coefficient": 0.03,
                              "time_gens_ago": 300})
    cases.append(("perfect plus 3 cold-spot calls", False, fp3))

    tmp = tempfile.mkdtemp()
    try:
        print(f"{'case':<36}{'expect':>8}{'got':>8}   verdict")
        bad = 0
        for name, expect_pass, sub in cases:
            got, last = run_verifier(truth, sub, tmp)
            verdict = "ok" if got == expect_pass else "*** WRONG ***"
            if got != expect_pass:
                bad += 1
            print(f"{name:<36}{str(expect_pass):>8}{str(got):>8}   {verdict}")
        print()
        print("all verifier self-tests behaved as intended" if bad == 0
              else f"{bad} verifier self-test(s) misbehaved")
        return 1 if bad else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
