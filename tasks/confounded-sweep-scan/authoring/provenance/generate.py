#!/usr/bin/env python3
"""Generate the shipped replicate: agent-facing data plus the sealed truth.

Usage:  generate.py OUTDIR [--max-replicates N] [--quick]

Writes into OUTDIR:
    public/variants.vcf.gz      agent-facing genotypes
    public/recombination_map.bed  agent-facing rate map
    public/README_DATA.txt      agent-facing description of the files
    sealed/truth.json           ground truth for the verifier
    provenance.json             seeds, versions, realized trajectories

Nothing in public/ depends on or reveals anything in sealed/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
import sweeps as S
from export import scrub_metadata, subsample, write_vcf
from forward import run_forward
from datadoc import render as render_data_readme
from genome import write_rate_map_bed
from recap import overlay_mutations, recapitate


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--max-replicates", type=int, default=12)
    ap.add_argument("--forward-only", action="store_true",
                    help="run the forward phase, print the realized sweeps "
                         "and stop; for tuning origins IN CONTEXT, which is "
                         "the only place interference between the ten sweeps "
                         "and the real genome length are present")
    ap.add_argument("--no-accept", action="store_true",
                    help="skip replicate acceptance (plumbing smoke test only)")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="replicate index, for independent probe runs")
    ap.add_argument("--max-attempts", type=int, default=800,
                    help="establishment attempts per sweep before giving up; "
                         "lower it on a diagnostic probe so a sweep that is "
                         "failing systematically reports its cause in minutes "
                         "rather than after the full budget")
    ap.add_argument("--stage-report-every", type=int, default=50,
                    help="print the accumulated establishment-failure causes "
                         "for a sweep every this many consecutive failures")
    ap.add_argument("--quick", action="store_true",
                    help="small smoke-test scale, not the shipped replicate")
    args = ap.parse_args()

    cfg = C.genome_config()
    demog = C.demography()
    t_forward = C.T_FORWARD
    n_cur = C.N_CURRENT

    if args.quick:
        cfg.n_chrom = 2
        cfg.chrom_len = 4_000_000
        cfg.cold_spots = [c for c in cfg.cold_spots if c.chrom < 2 and c.end < 4_000_000]
        n_cur = 1500
        t_forward = 600

    pub = os.path.join(args.outdir, "public")
    seal = os.path.join(args.outdir, "sealed")
    os.makedirs(pub, exist_ok=True)
    os.makedirs(seal, exist_ok=True)

    if args.quick:
        # The shipped layout is designed for 12 Mb chromosomes; reusing it on a
        # shrunken genome violates the spacing constraints. Quick mode gets its
        # own miniature zoo, purely to smoke-test the pipeline end to end.
        from forward import SweepPlan
        plans = [
            SweepPlan("sw01", 2_013_457, 0.05, 1.0, 300, 1, "hard_complete"),
            SweepPlan("sw06", 5_907_311, 0.05, 1.0, 400, 20, "soft_complete"),
            SweepPlan("sw09", 6_803_229, 0.03, 1.0, 260, 1, "hard_partial"),
        ]
    else:
        plans = S.build_plans(cfg, C.MASTER_SEED)

    if args.forward_only:
        seed = C.MASTER_SEED + 1000 * args.seed_offset
        t0 = time.time()
        _, rec, el = run_forward(cfg, n_cur, t_forward, plans, seed=seed,
                                 traj_thin=10, accept_fn=None,
                                 max_attempts_per_stage=args.max_attempts,
                                 stage_report_every=args.stage_report_every)
        ok, problems = S.check_acceptance(rec.real)
        print(f"\nforward {el:.0f}s  seed {seed}  accepted={ok}")
        print(f"{'id':<6}{'scheduled':<16}{'realized':<16}{'s':>8}"
              f"{'origin':>8}{'done_ago':>10}{'freq':>8}{'origins':>9}"
              f"{'attempts':>10}")
        for sid in sorted(rec.real):
            r = rec.real[sid]
            print(f"{sid:<6}{r.intended_class:<16}{S.realized_class(r):<16}"
                  f"{r.s:>8.4f}{r.origin_gens_ago:>8}"
                  f"{str(r.completion_gens_ago if r.completed else '-'):>10}"
                  f"{r.final_freq:>8.3f}{r.n_origins_surviving:>9}"
                  f"{r.establishment_attempts:>10}")
        for pr in problems:
            print(f"  - {pr}")
        return

    accepted = None
    attempts_log = []
    for rep in range(args.max_replicates):
        seed = C.MASTER_SEED + 1000 * rep
        print(f"\n=== replicate {rep} (seed {seed}) ===", flush=True)
        t0 = time.time()
        ts, rec, el = run_forward(cfg, n_cur, t_forward, plans, seed=seed,
                                  traj_thin=10,
                                  max_attempts_per_stage=args.max_attempts,
                                  stage_report_every=args.stage_report_every,
                                  accept_fn=None if args.no_accept
                                  else S.failing_sweeps)
        ok, problems = S.check_acceptance(rec.real)
        if args.no_accept:
            problems = [f"(ignored) {x}" for x in problems]
            ok = True
        attempts_log.append({"replicate": rep, "seed": seed,
                             "forward_seconds": round(el, 1),
                             "accepted": ok, "problems": problems})
        print(f"  forward {el:.0f}s; accepted={ok}", flush=True)
        for p in problems:
            print(f"    - {p}", flush=True)
        if ok:
            accepted = (rep, seed, ts, rec)
            break
    if accepted is None:
        print("\nNO REPLICATE ACCEPTED", file=sys.stderr)
        json.dump(attempts_log, open(os.path.join(args.outdir, "attempts.json"), "w"), indent=2)
        sys.exit(2)

    rep, seed, ts, rec = accepted
    print(f"\nusing replicate {rep} (seed {seed})", flush=True)

    t0 = time.time()
    rts = recapitate(ts, cfg, demog, seed=seed + 1)
    print(f"recapitated in {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    mts = overlay_mutations(rts, cfg, seed=seed + 2)
    print(f"mutations overlaid in {time.time()-t0:.0f}s: {mts.num_sites} sites", flush=True)

    sts, chosen = subsample(mts, C.N_SAMPLE_DIPLOIDS, seed=seed + 3)
    sts = scrub_metadata(sts)
    print(f"subsampled to {C.N_SAMPLE_DIPLOIDS} diploids: {sts.num_sites} sites", flush=True)

    vcf_path = os.path.join(pub, "variants.vcf.gz")
    t0 = time.time()
    stats = write_vcf(sts, cfg, vcf_path)
    print(f"wrote VCF in {time.time()-t0:.0f}s: {stats}", flush=True)

    bed_path = os.path.join(pub, "recombination_map.bed")
    write_rate_map_bed(cfg, bed_path)

    # Rendered from the config rather than hand-maintained, so the file that
    # tells the agent how many individuals and chromosomes there are cannot
    # disagree with the data beside it.
    doc_path = os.path.join(pub, "README_DATA.txt")
    with open(doc_path, "w") as fh:
        fh.write(render_data_readme(C.N_SAMPLE_DIPLOIDS, cfg.n_chrom,
                                    cfg.chrom_len, cfg.mut_rate))
    print(f"wrote {doc_path}", flush=True)

    # ---------------- sealed truth ----------------
    truth = {
        "genome": {
            "n_chromosomes": cfg.n_chrom,
            "chromosome_length_bp": cfg.chrom_len,
            "base_recombination_rate": cfg.base_rec_rate,
            "mutation_rate": cfg.mut_rate,
        },
        "demography": demog.to_dict(),
        "forward_phase_generations": t_forward,
        "n_sample_diploids": C.N_SAMPLE_DIPLOIDS,
        "cold_spots": [
            {"chrom": c.chrom + 1,
             "start": c.start, "end": c.end, "factor": c.factor}
            for c in cfg.cold_spots
        ],
        "sweeps": [],
    }
    for sid in sorted(rec.real):
        r = rec.real[sid]
        c = r.position // cfg.chrom_len
        truth["sweeps"].append({
            "sweep_id": sid,
            "chrom": c + 1,
            "position_global": r.position,
            "position_local": r.position - cfg.chrom_offset(c),
            "selection_coefficient": r.s,
            "dominance": r.h,
            "class": S.realized_class(r),
            "intended_class": r.intended_class,
            "n_origins_introduced": r.n_origins_introduced,
            "n_origins_surviving": r.n_origins_surviving,
            "origin_gens_ago": r.origin_gens_ago,
            "completion_gens_ago": r.completion_gens_ago if r.completed else None,
            "fixation_gens_ago": r.fixation_gens_ago if r.fixed else None,
            "completed": r.completed,
            "final_frequency": round(r.final_freq, 6),
            "fixed": r.fixed,
        })
    with open(os.path.join(seal, "truth.json"), "w") as fh:
        json.dump(truth, fh, indent=2, sort_keys=True)

    # ---------------- provenance ----------------
    import fwdpy11, msprime, tskit
    prov = {
        "master_seed": C.MASTER_SEED,
        "accepted_replicate": rep,
        "accepted_seed": seed,
        "attempts": attempts_log,
        "versions": {
            "python": platform.python_version(),
            "fwdpy11": fwdpy11.__version__,
            "msprime": msprime.__version__,
            "tskit": tskit.__version__,
            "numpy": np.__version__,
        },
        "export_stats": stats.__dict__,
        "sha256": {
            "variants.vcf.gz": sha256(vcf_path),
            "recombination_map.bed": sha256(bed_path),
        },
        "realized": {sid: rec.real[sid].to_dict(include_trajectory=False)
                     for sid in sorted(rec.real)},
    }
    with open(os.path.join(args.outdir, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2, sort_keys=True)

    # trajectories kept separately: large, and authoring-only
    with open(os.path.join(args.outdir, "trajectories.json"), "w") as fh:
        json.dump({sid: rec.real[sid].to_dict(include_trajectory=True)
                   for sid in sorted(rec.real)}, fh)

    print("\nDONE")
    # Print the GRADED time field. Exact fixation lags completion by
    # hundreds of generations when a sweep has several origins, so printing
    # fixation here reads as though acceptance had been violated.
    print(f"{'id':<6}{'class':<22}{'done_ago':>9}{'freq':>7}{'origins':>8}")
    for s in truth["sweeps"]:
        print(f"{s['sweep_id']:<6}{s['class']:<22}"
              f"{str(s['completion_gens_ago']):>9}"
              f"{s['final_frequency']:>7.3f}{s['n_origins_surviving']:>8}")


if __name__ == "__main__":
    main()
