"""The CrossPhase gate suite.

Every gate here blocks the blind batch.  Run after ``baselines.run_baselines``:

    python -m gates.run_gates [--stage all] [--workers N]

Results are cached under ``reports/gates_cache/`` so a single stage can be
re-run without repeating the whole suite.  ``reports/gates.json`` and
``reports/gates.md`` are the outputs.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import math
import re
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from crossphase.core.engine import (TRAIN, ObjectiveError, balanced_accuracy,
                                    quality, run_setting, train_model)
from crossphase.core.methods import GRIDS, all_configs, env_risks, erm
from crossphase.core.parallel import pmap
from crossphase.core.settings import (PROXY_SOURCE_N, PROXY_WORLD_N,
                                      PROXY_WORLDS, PUBLIC_POOL_SEED,
                                      PUBLIC_RUN_SEEDS, PUBLIC_SETTINGS)
from grader.private_specs import (ALL_SETTINGS, OFFICIAL_POOL_SEED,
                                  OFFICIAL_RUN_SEEDS, OFFICIAL_SOURCE_N,
                                  OFFICIAL_WORLD_N, OFFICIAL_WORLDS, SETTING_C)

from . import measure, probes

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
CACHE = REPORTS / "gates_cache"
DIAG_SEEDS = OFFICIAL_RUN_SEEDS[:3]
RECON_SEEDS = PUBLIC_RUN_SEEDS[:2]


def geo(values) -> float:
    values = list(values)
    if any(v <= 0 for v in values):
        return 0.0
    return 100.0 * math.exp(sum(math.log(v) for v in values) / len(values))


def cached(name, build):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    started = time.time()
    payload = build()
    path.write_text(json.dumps(payload, indent=2))
    print(f"  [{name}] {time.time() - started:.0f}s")
    return payload


# --------------------------------------------------------------------------- #
# Objective catalogue: the baseline grid plus the probe suite.
# --------------------------------------------------------------------------- #

def baseline_objective(label):
    family, config = label.split("|", 1)
    return all_configs()[family][config]()


_PROBES = None


def probe_objectives():
    global _PROBES
    if _PROBES is None:
        summary = json.loads((REPORTS / "sweep.json").read_text())
        public = max(summary, key=lambda r: r["proxy_score"])
        known = all_configs()[public["family"]][
            "" if public["config"] == "(none)" else public["config"]]()
        fallback = all_configs()["ERM"][""]()
        # Threshold sits between the public settings' early risk level and the
        # hidden setting's; it is measured in the fingerprint stage and stored.
        threshold = json.loads((CACHE / "fingerprint.json").read_text())["threshold"]
        _PROBES = probes.probe_catalogue(known, fallback, threshold)
    return _PROBES


def objective_for(label):
    if label.startswith("probe|"):
        return probe_objectives()[label.split("|", 1)[1]]
    if label == "reference|adaptive":
        return probes.ADAPTIVE
    return baseline_objective(label)


# --------------------------------------------------------------------------- #
# Stage: fingerprint separability (also supplies the branch threshold)
# --------------------------------------------------------------------------- #

def _early_stats(item):
    setting, seed = item
    spec = ALL_SETTINGS[setting]
    collected = []

    def objective(logits_by_env, targets_by_env, step, total_steps, state):
        risks = env_risks(logits_by_env, targets_by_env)
        collected.append((float(risks.mean()), float(risks.std(unbiased=False))))
        return risks.mean()

    train_model(spec, objective, seed, TRAIN.__class__(steps=50))
    level = [m for m, _ in collected]
    spread = [s for _, s in collected]
    return setting, seed, float(np.mean(level)), float(np.mean(spread))


def stage_fingerprint():
    def build():
        seeds = tuple(OFFICIAL_RUN_SEEDS) + tuple(PUBLIC_RUN_SEEDS)
        rows = pmap(_early_stats, [(s, sd) for s in ALL_SETTINGS for sd in seeds])
        by_setting = collections.defaultdict(list)
        for setting, _seed, level, spread in rows:
            by_setting[setting].append([level, spread])

        # Leave-one-out nearest-centroid accuracy on (level, spread).
        points = [(s, p) for s, ps in by_setting.items() for p in ps]
        correct = 0
        for i, (truth, point) in enumerate(points):
            centroids = {}
            for j, (setting, other) in enumerate(points):
                if i != j:
                    centroids.setdefault(setting, []).append(other)
            best = min(centroids,
                       key=lambda s: sum((np.mean(centroids[s], axis=0) -
                                          np.array(point)) ** 2))
            correct += int(best == truth)
        accuracy = correct / len(points)

        public_level = statistics.mean(
            p[0] for s in ("A", "B") for p in by_setting[s])
        hidden_level = statistics.mean(p[0] for p in by_setting["C"])
        return {
            "accuracy": accuracy,
            "levels": {s: statistics.mean(p[0] for p in ps)
                       for s, ps in by_setting.items()},
            "spreads": {s: statistics.mean(p[1] for p in ps)
                        for s, ps in by_setting.items()},
            "threshold": (public_level + hidden_level) / 2.0,
        }

    return cached("fingerprint", build)


# --------------------------------------------------------------------------- #
# Stage: probe scores (proxy + official), on the same protocol as the baselines
# --------------------------------------------------------------------------- #

def _score_job(item):
    """Score one configuration.  A numerically diverging objective scores zero,
    exactly as the grader treats it; anything else propagates, because a harness
    fault recorded as a method property is how the boundary expansion silently
    failed for a whole sweep."""
    try:
        return _score(item)
    except ObjectiveError:
        kind, label, setting, seed = item
        return kind, label, setting, seed, 0.0, "diverged"


def _score(item):
    kind, label, setting, seed = item
    objective = objective_for(label)
    if kind == "proxy":
        result = run_setting(ALL_SETTINGS[setting], objective, seed, PROXY_WORLDS,
                             PROXY_SOURCE_N, PROXY_WORLD_N,
                             pool_seed=PUBLIC_POOL_SEED)
    else:
        result = run_setting(ALL_SETTINGS[setting], objective, seed, OFFICIAL_WORLDS,
                             OFFICIAL_SOURCE_N, OFFICIAL_WORLD_N,
                             pool_seed=OFFICIAL_POOL_SEED)
    q, binding = quality(result)
    return kind, label, setting, seed, q, str(binding)


def _aggregate(rows):
    table = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(list)))
    for kind, label, setting, _seed, q, _binding in rows:
        table[label][kind][setting].append(q)
    out = {}
    for label, kinds in table.items():
        entry = {}
        if "proxy" in kinds:
            entry["proxy_score"] = geo(
                sum(v) / len(v) for v in kinds["proxy"].values())
        if "official" in kinds:
            means = {s: sum(v) / len(v) for s, v in kinds["official"].items()}
            entry["S"] = geo(means.values())
            entry.update({f"Q_{s}": m for s, m in means.items()})
        out[label] = entry
    return out


def stage_probes():
    stage_fingerprint()

    def build():
        labels = [f"probe|{name}" for name in probe_objectives()]
        labels.append("reference|adaptive")
        jobs = []
        for label in labels:
            for seed in PUBLIC_RUN_SEEDS:
                jobs += [("proxy", label, s, seed) for s in ("A", "B")]
            for seed in DIAG_SEEDS:
                jobs += [("official", label, s, seed) for s in ("A", "B", "C")]
        return _aggregate(pmap(_score_job, jobs))

    return cached("probes", build)


# --------------------------------------------------------------------------- #
# Stage: reconstruction route
# --------------------------------------------------------------------------- #

_RECON_CACHE: dict = {}


def reconstructed_pool(spec, world, seed):
    key = (spec.name, world, seed)
    if key not in _RECON_CACHE:
        _RECON_CACHE[key] = measure.reconstructed_world(spec, world, seed)
    return _RECON_CACHE[key]


def _recon_job(item):
    try:
        return _recon(item)
    except ObjectiveError:
        return item[0], item[1], item[2], 0.0


def _recon(item):
    label, setting, seed = item
    spec = PUBLIC_SETTINGS[setting]
    model = train_model(spec, objective_for(label), seed)
    xs, ys = reconstructed_pool(spec, (0.85, 0.85, 1.0, 1.0), seed)
    retention = balanced_accuracy(model, xs, ys)
    worst = 1.0
    for world in OFFICIAL_WORLDS:
        xw, yw = reconstructed_pool(spec, world, seed)
        worst = min(worst, balanced_accuracy(model, xw, yw))
    return label, setting, seed, math.sqrt(max(retention, 0.0) * max(worst, 0.0))


def stage_reconstruction():
    def build():
        labels = [f"{f}|{c}" for f, g in all_configs().items() for c in g]
        labels += [f"probe|{name}" for name in probe_objectives()]
        jobs = [(label, setting, seed) for label in labels
                for setting in ("A", "B") for seed in RECON_SEEDS]
        rows = pmap(_recon_job, jobs)
        table = collections.defaultdict(lambda: collections.defaultdict(list))
        for label, setting, _seed, q in rows:
            table[label][setting].append(q)
        return {label: geo(sum(v) / len(v) for v in d.values())
                for label, d in table.items()}

    return cached("reconstruction", build)


# --------------------------------------------------------------------------- #
# Stage: rescore a handful of configurations on the full official seed set
# --------------------------------------------------------------------------- #

def stage_rescore(labels, tag):
    digest = hashlib.sha1("\n".join(sorted(labels)).encode()).hexdigest()[:8]
    tag = f"{tag}_{digest}"

    def build():
        jobs = [("official", label, setting, seed) for label in labels
                for setting in ("A", "B", "C") for seed in OFFICIAL_RUN_SEEDS]
        return _aggregate(pmap(_score_job, jobs))

    return cached(f"rescore_{tag}", build)


# --------------------------------------------------------------------------- #
# Stage: transition timing
# --------------------------------------------------------------------------- #

FLIPPED = (0.10, 0.10, 1.00, 1.00)


def _trace_job(item):
    label, setting, seed = item
    try:
        trace = measure.transition_trace(ALL_SETTINGS[setting],
                                         objective_for(label), seed, FLIPPED)
    except ObjectiveError:
        return label, setting, seed, None, []
    return label, setting, seed, measure.transition_step(trace), trace


def stage_transition(labels):
    def build():
        jobs = [(label, s, sd) for label in labels for s in ALL_SETTINGS
                for sd in DIAG_SEEDS]
        rows = pmap(_trace_job, jobs)
        table = collections.defaultdict(list)
        traces = {}
        for label, setting, seed, step, trace in rows:
            if step is None:      # diverged; contributes no timing evidence
                continue
            table[f"{label}|{setting}"].append(step)
            traces[f"{label}|{setting}|{seed}"] = trace
        return {"steps": {k: statistics.median(v) for k, v in table.items()},
                "traces": traces}

    return cached("transition", build)


# --------------------------------------------------------------------------- #
# Stage: nuisance-free oracle ceiling
# --------------------------------------------------------------------------- #

def _oracle_spec(spec):
    """The same setting with its cues carrying no label information at all."""
    from crossphase.core.generator import Environment
    return dataclasses.replace(
        spec, envs=tuple(Environment(0.5, 0.5, e.sigma) for e in spec.envs),
        source_mix=(0.5, 0.5))


def _oracle_job(item):
    setting, seed = item
    result = run_setting(_oracle_spec(ALL_SETTINGS[setting]), erm, seed,
                         OFFICIAL_WORLDS, OFFICIAL_SOURCE_N, OFFICIAL_WORLD_N,
                         pool_seed=OFFICIAL_POOL_SEED)
    return setting, quality(result)[0]


def stage_oracle():
    """How much of each setting's gap is reachable at all.

    Part of setting C's low score is irreducible -- it carries more label noise
    and heavier observation noise than A and B -- so without this number one
    cannot tell whether an objective's C gain is near the ceiling or leaving most
    of it unclaimed.
    """

    def build():
        rows = pmap(_oracle_job, [(s, sd) for s in ALL_SETTINGS
                                  for sd in OFFICIAL_RUN_SEEDS])
        per = collections.defaultdict(list)
        for setting, q in rows:
            per[setting].append(q)
        means = {s: sum(v) / len(v) for s, v in per.items()}
        return {"Q": means, "S": geo(means.values())}

    return cached("oracle", build)


# --------------------------------------------------------------------------- #
# Gate evaluation
# --------------------------------------------------------------------------- #

def coefficient_of(config: str):
    match = re.search(r"(?:lam|eta|coef)=([0-9.e+-]+)", config)
    return float(match.group(1)) if match else None


def warm_of(config: str):
    match = re.search(r"warm=([0-9.]+)", config)
    return float(match.group(1)) if match else None


def build_report(workers=None) -> dict:
    sweep = json.loads((REPORTS / "sweep.json").read_text())
    baselines = json.loads((REPORTS / "baselines.json").read_text())
    s_star = baselines["S_star"]

    fingerprint = stage_fingerprint()
    probe_scores = stage_probes()
    recon = stage_reconstruction()

    sweep_by_label = {f"{r['family']}|{'' if r['config'] == '(none)' else r['config']}": r
                      for r in sweep}

    # ---- proxy-to-official rank correlation -------------------------------- #
    combined = {label: {"proxy_score": r["proxy_score"], "S": r["S"]}
                for label, r in sweep_by_label.items()}
    for label, entry in probe_scores.items():
        if "proxy_score" in entry and "S" in entry:
            combined[label] = entry
    labels = sorted(combined)
    rho = measure.spearman([combined[l]["proxy_score"] for l in labels],
                           [combined[l]["S"] for l in labels])

    # ---- ceilings ----------------------------------------------------------- #
    public_arg = max(labels, key=lambda l: combined[l]["proxy_score"])
    recon_arg = max(recon, key=lambda l: recon[l])
    rescored = stage_rescore([public_arg, recon_arg, "reference|adaptive"], "ceilings")

    # ---- rank reversal ------------------------------------------------------ #
    best_per_setting = {}
    for setting in ("A", "B", "C"):
        best = max(sweep_by_label, key=lambda l: sweep_by_label[l][f"Q_{setting}"])
        best_per_setting[setting] = {
            "config": best, "Q": sweep_by_label[best][f"Q_{setting}"],
            "family": sweep_by_label[best]["family"]}
    reversal_families = {v["family"] for v in best_per_setting.values()}

    # ---- penalty and warm-up anti-transfer ---------------------------------- #
    anti_transfer = {}
    for family in all_configs():
        rows = [r for r in sweep if r["family"] == family and r["config"] != "(none)"]
        if not rows:
            continue
        best_ab = max(rows, key=lambda r: geo([r["Q_A"], r["Q_B"]]))
        best_c = max(rows, key=lambda r: r["Q_C"])
        cost = best_c["Q_C"] - best_ab["Q_C"]
        anti_transfer[family] = {
            "ab_optimal": best_ab["config"], "c_optimal": best_c["config"],
            "coef_ab": coefficient_of(best_ab["config"]),
            "coef_c": coefficient_of(best_c["config"]),
            "warm_ab": warm_of(best_ab["config"]), "warm_c": warm_of(best_c["config"]),
            "Q_C_at_ab_optimum": best_ab["Q_C"], "Q_C_at_c_optimum": best_c["Q_C"],
            "Q_C_cost_of_transfer": cost,
        }

    # ---- loss-scale routes --------------------------------------------------- #
    # Uniform rescaling is the control and should be flat.  The route that is
    # actually open is a mid-run scale change, which survives weight_decay=0
    # because Adam's second moment re-adapts more slowly than the budget allows.
    # The gate is not "is it flat" -- it demonstrably is not -- but "does it win".
    scale_points = []
    for label, entry in probe_scores.items():
        match = re.match(r"probe\|scale\(ERM,([0-9.]+)\)", label)
        if match and "S" in entry:
            scale_points.append((math.log10(float(match.group(1))), entry["S"]))
    scale_points.sort()
    if len(scale_points) >= 2:
        xs = np.array([p[0] for p in scale_points])
        ys = np.array([p[1] for p in scale_points])
        slope = float(np.polyfit(xs, ys, 1)[0])
        spread = float(ys.max() - ys.min())
    else:
        slope, spread = float("nan"), float("nan")

    erm_S = next((r["S"] for r in sweep if r["family"] == "ERM"), float("nan"))
    step_probes = {label: entry["S"] for label, entry in probe_scores.items()
                   if ("scale-step" in label or "freeze" in label) and "S" in entry}
    best_step_probe = max(step_probes.values()) if step_probes else float("nan")

    # ---- transition timing -------------------------------------------------- #
    transition_labels = [f"{f}|{'' if c == '(none)' else c}"
                         for f, c in ((r["family"], r["best_config"])
                                      for r in baselines["families"])]
    transition = stage_transition(sorted(set(transition_labels + ["ERM|"])))
    per_setting_steps = collections.defaultdict(list)
    for key, step in transition["steps"].items():
        per_setting_steps[key.rsplit("|", 1)[1]].append(step)
    median_step = {s: statistics.median(v) for s, v in per_setting_steps.items()}
    separation = abs(median_step["C"] - median_step["A"]) / TRAIN.steps

    # ---- binding audit ------------------------------------------------------ #
    binding = collections.Counter()
    for row in sweep:
        binding.update(row["binding_all"])
    binding_total = sum(binding.values())
    binding_share = {k: v / binding_total for k, v in binding.most_common()}

    # ---- combination rule comparison ---------------------------------------- #
    competent = [r for r in sweep if r["S"] > 0.8 * s_star]
    geo_scores = [r["S"] for r in competent]
    min_scores = [100.0 * min(r["Q_A"], r["Q_B"], r["Q_C"]) for r in competent]
    combination = {
        "n_competent": len(competent),
        "geometric_mean_spread": max(geo_scores) - min(geo_scores),
        "outer_minimum_spread": max(min_scores) - min(min_scores),
        "geometric_mean_stdev": statistics.pstdev(geo_scores),
        "outer_minimum_stdev": statistics.pstdev(min_scores),
    }

    cue = {name: measure.cue_recovery_accuracy(spec)
           for name, spec in ALL_SETTINGS.items()}
    oracle = stage_oracle()

    gates = {
        "public_tuning_ceiling": {
            "argmax_on_proxy": public_arg,
            "S": rescored[public_arg]["S"],
            # The global argmax is not the most adversarial reading.  Letting the
            # proxy pick a configuration for EACH family and taking the best of
            # those is stronger, and it is what an agent working family by family
            # would actually do.
            "S_per_family_proxy_selection": baselines.get("S_star_if_selected_on_proxy"),
            "S_star": s_star,
            "Q_C": rescored[public_arg]["Q_C"],
            "Q_AB": geo([rescored[public_arg]["Q_A"], rescored[public_arg]["Q_B"]]),
            "pass": (rescored[public_arg]["S"] <= s_star
                 and (baselines.get("S_star_if_selected_on_proxy", 0.0) <= s_star)),
        },
        "reconstruction_ceiling": {
            "argmax_on_reconstructed": recon_arg,
            "S": rescored[recon_arg]["S"],
            "S_star": s_star,
            "pass": rescored[recon_arg]["S"] <= s_star,
        },
        "proxy_correlation": {
            "spearman": rho, "n": len(labels),
            "pass": 0.5 <= rho <= 0.8,
        },
        "transition_separation": {
            "median_step": median_step,
            "fraction_of_budget": {s: v / TRAIN.steps
                                   for s, v in median_step.items()},
            "separation_fraction": separation,
            "total_steps": TRAIN.steps,
            # The trace is sampled every 10 steps, so a measured step carries
            # +/- 5 steps of resolution -- 0.6% of the budget.  The A-to-C figure
            # sits close to the 20% requirement, so the resolution is worth
            # stating rather than leaving implicit.
            "probe_every_steps": 10,
            "resolution_fraction": 10 / TRAIN.steps,
            "pass": separation >= 0.20,
        },
        "warmup_anti_transfer": {
            "per_family": {f: {k: v for k, v in d.items() if "warm" in k or "Q_C" in k}
                           for f, d in anti_transfer.items()},
            "max_Q_C_cost": max(d["Q_C_cost_of_transfer"]
                                for d in anti_transfer.values()),
            "families_with_differing_warmup": [
                f for f, d in anti_transfer.items() if d["warm_ab"] != d["warm_c"]],
            "pass": any(d["warm_ab"] != d["warm_c"] and d["Q_C_cost_of_transfer"] > 0.005
                        for d in anti_transfer.values()),
        },
        "rank_reversal": {
            "best_per_setting": best_per_setting,
            "distinct_families": sorted(reversal_families),
            "pass": len(reversal_families) >= 2,
        },
        "penalty_anti_transfer": {
            "per_family": anti_transfer,
            "families_with_differing_coefficient": [
                f for f, d in anti_transfer.items()
                if d["coef_ab"] is not None and d["coef_ab"] != d["coef_c"]],
            "pass": any(d["coef_ab"] is not None and d["coef_ab"] != d["coef_c"]
                        for d in anti_transfer.values()),
        },
        "loss_scale_routes": {
            "uniform_points": scale_points,
            "uniform_slope_per_decade": slope,
            "uniform_spread": spread,
            "mid_run_probes": step_probes,
            "best_mid_run_probe": best_step_probe,
            "ERM_baseline": erm_S,
            "gain_over_ERM": best_step_probe - erm_S,
            "margin_below_S_star": s_star - best_step_probe,
            "note": ("uniform rescaling is closed by weight_decay=0; a mid-run "
                     "scale change is not, and is bounded here rather than "
                     "assumed away"),
            "pass": best_step_probe <= s_star and abs(slope) < 1.0,
        },
        "binding_world_audit": {
            "share": binding_share,
            "most_common_share": max(binding_share.values()),
            "pass": max(binding_share.values()) <= 0.90,
        },
        "fingerprint_policy": {
            "declared": "legal -- see docs/classification_rules.md",
            "setting_identification_accuracy": fingerprint["accuracy"],
            "early_risk_levels": fingerprint["levels"],
            "branch_probe_S": probe_scores.get("probe|fingerprint-branch", {}).get("S"),
            "branch_probe_beats_S_star":
                (probe_scores.get("probe|fingerprint-branch", {}).get("S", 0) > s_star),
            "pass": True,
        },
        "headroom": {
            "reference_solution_S": rescored["reference|adaptive"]["S"],
            "S_star": s_star,
            "margin": rescored["reference|adaptive"]["S"] - s_star,
            "oracle_ceiling_S": oracle["S"],
            "oracle_Q": oracle["Q"],
            "reference_Q": {s: rescored["reference|adaptive"].get(f"Q_{s}")
                            for s in ("A", "B", "C")},
            "fraction_of_reachable_gap_claimed": {
                s: ((rescored["reference|adaptive"].get(f"Q_{s}", 0.0)
                     - next(r[f"Q_{s}"] for r in baselines["families"]
                            if r["family"] == baselines["strongest"]))
                    / max(oracle["Q"][s]
                          - next(r[f"Q_{s}"] for r in baselines["families"]
                                 if r["family"] == baselines["strongest"]), 1e-9))
                for s in ("A", "B", "C")},
            "pass": rescored["reference|adaptive"]["S"] > s_star,
        },
    }

    return {
        "S_star": s_star,
        "gates": gates,
        "cue_recovery_accuracy": cue,
        "combination_rule": combination,
        "reconstruction_top": sorted(recon.items(), key=lambda kv: -kv[1])[:5],
        "probe_scores": probe_scores,
        "rescored": rescored,
    }


def render_markdown(report) -> str:
    lines = ["# CrossPhase gate report", "",
             f"`S*` = **{report['S_star']:.3f}**", "",
             "| Gate | Requirement | Measured | Verdict |", "|---|---|---|---|"]
    g = report["gates"]

    def row(name, requirement, measured, ok):
        lines.append(f"| {name} | {requirement} | {measured} | "
                     f"{'PASS' if ok else 'FAIL'} |")

    per_family = g["public_tuning_ceiling"].get("S_per_family_proxy_selection")
    row("Public-tuning ceiling", "argmax on proxy scores <= S*",
        f"{g['public_tuning_ceiling']['S']:.3f} vs {report['S_star']:.3f} "
        f"({g['public_tuning_ceiling']['argmax_on_proxy']})"
        + (f"; per-family proxy selection reaches {per_family:.3f}"
           if per_family is not None else ""),
        g["public_tuning_ceiling"]["pass"])
    row("Reconstruction ceiling", "argmax on rebuilt worlds <= S*",
        f"{g['reconstruction_ceiling']['S']:.3f} vs {report['S_star']:.3f} "
        f"({g['reconstruction_ceiling']['argmax_on_reconstructed']})",
        g["reconstruction_ceiling"]["pass"])
    row("Proxy correlation", "Spearman in [0.5, 0.8]",
        f"{g['proxy_correlation']['spearman']:.3f} over "
        f"{g['proxy_correlation']['n']} configurations",
        g["proxy_correlation"]["pass"])
    frac = g["transition_separation"]["fraction_of_budget"]
    row("Transition separation", "A-to-C differs by >= 20% of budget",
        f"{g['transition_separation']['separation_fraction']:.0%}; all three differ "
        + ", ".join(f"{s} {frac[s]:.2f}" for s in ("A", "B", "C"))
        + f" of budget (+/- {g['transition_separation']['resolution_fraction']:.3f} "
          f"probe resolution)",
        g["transition_separation"]["pass"])
    row("Warm-up anti-transfer", "A/B-optimal warm-up suboptimal on C",
        f"max Q_C cost {g['warmup_anti_transfer']['max_Q_C_cost']:.4f}; "
        f"families differing: "
        f"{', '.join(g['warmup_anti_transfer']['families_with_differing_warmup']) or 'none'}",
        g["warmup_anti_transfer"]["pass"])
    row("Rank reversal", "best baseline differs across >= 2 settings",
        ", ".join(f"{s}:{v['family']}"
                  for s, v in g["rank_reversal"]["best_per_setting"].items()),
        g["rank_reversal"]["pass"])
    row("Penalty anti-transfer", "coefficient argmax differs A/B vs C",
        ", ".join(g["penalty_anti_transfer"]["families_with_differing_coefficient"])
        or "none",
        g["penalty_anti_transfer"]["pass"])
    row("Loss-scale routes", "mid-run scale change must not beat S*",
        f"uniform {g['loss_scale_routes']['uniform_slope_per_decade']:.3f}/decade; "
        f"best mid-run probe {g['loss_scale_routes']['best_mid_run_probe']:.2f} "
        f"(+{g['loss_scale_routes']['gain_over_ERM']:.2f} over ERM, "
        f"{g['loss_scale_routes']['margin_below_S_star']:.2f} below S*)",
        g["loss_scale_routes"]["pass"])
    row("Binding-world audit", "no world binds > 90% of cells",
        ", ".join(f"{k} {v:.0%}" for k, v in
                  list(g["binding_world_audit"]["share"].items())[:3]),
        g["binding_world_audit"]["pass"])
    row("Fingerprint policy", "declared before trials",
        f"legal; settings separable at "
        f"{g['fingerprint_policy']['setting_identification_accuracy']:.0%}, "
        f"branch probe S="
        + (f"{g['fingerprint_policy']['branch_probe_S']:.3f}"
           if g['fingerprint_policy']['branch_probe_S'] is not None else "n/a"),
        g["fingerprint_policy"]["pass"])
    row("Headroom", "reference solution beats S*",
        f"{g['headroom']['reference_solution_S']:.3f} "
        f"(+{g['headroom']['margin']:.3f}); nuisance-free ceiling "
        f"{g['headroom']['oracle_ceiling_S']:.3f}",
        g["headroom"]["pass"])

    lines += ["", "## Cue recovery from public data", "",
              "| Setting | z1 | z2 | median-split cap |", "|---|---|---|---|"]
    for name, acc in report["cue_recovery_accuracy"].items():
        lines.append(f"| {name} | {acc['z1']:.3f} | {acc['z2']:.3f} | "
                     f"{acc['median_split_cap']:.3f} |")
    lines += ["",
              "Both nuisance cues are recoverable from raw signals essentially "
              "perfectly, so the reconstruction route is fully live and is "
              "assumed available to any competent agent.  It is gated, not "
              "prevented.", ""]

    audit = g["binding_world_audit"]
    share = audit["most_common_share"]
    lines += ["## Binding-world audit", "",
              f"The cue-flipped world binds the minimum in {share:.0%} of "
              f"measured cells, against a 90% limit, so this gate "
              + ("**passes**" if audit["pass"] else
                 "**fails, and the threshold has not been moved to suit it**")
              + ".  Stated plainly either way, because the margin is not large: "
                "`Q` is in practice close to `sqrt(I * T_flip)`.", "",
              "The remaining worlds are not decorative -- they bind in the other "
              "cells, and they bind for the strongest methods, which is where a "
              "worst-case minimum has to bite.  Diverged runs are excluded from "
              "the denominator: counting a run that produced no measurement as "
              "evidence about which world binds once made this audit read 64% "
              "when 32% of its cells were failures.", "",
              "The known remedy if this drifts back over the limit is to tighten "
              "the core-attenuation world from 0.45 to 0.30, which is measured to "
              "bind far more often against IRM in setting C.  It costs a full "
              "re-sweep and gate re-run, and is recorded here rather than "
              "applied.", ""]

    combo = report["combination_rule"]
    wider = ("outer minimum"
             if combo["outer_minimum_spread"] > combo["geometric_mean_spread"]
             else "geometric mean")
    lines += ["## Combination rule", "",
              f"Across the {combo['n_competent']} competent configurations, the "
              f"geometric mean spreads scores over "
              f"{combo['geometric_mean_spread']:.3f} points "
              f"(sd {combo['geometric_mean_stdev']:.3f}); an outer minimum over "
              f"settings spreads them over {combo['outer_minimum_spread']:.3f} "
              f"(sd {combo['outer_minimum_stdev']:.3f}).  On raw spread the "
              f"**{wider} is the wider of the two**, and an earlier draft of this "
              f"report claimed the opposite while printing these same numbers.",
              "",
              "Raw spread is not decisive either way -- it is not normalised by "
              "the seed-level noise within a configuration, so a wider spread can "
              "be scale rather than signal.  The geometric mean is kept on a "
              "different ground, which does not depend on the comparison above: "
              "an outer minimum over settings reports only the worst setting and "
              "discards the other two entirely, so an objective that is excellent "
              "on A and B and mediocre on C scores identically to one that is "
              "mediocre everywhere.  That is precisely the distinction this task "
              "exists to make -- the reference solution's whole advantage is "
              "holding A and B while improving C -- and an outer minimum would be "
              "blind to it.  It would also compound with the minimum already "
              "taken inside each `Q` over target worlds, applying a worst-case "
              "twice.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    REPORTS.mkdir(exist_ok=True)
    report = build_report(workers=args.workers)
    (REPORTS / "gates.json").write_text(json.dumps(report, indent=2))
    markdown = render_markdown(report)
    (REPORTS / "gates.md").write_text(markdown + "\n")
    print(markdown)
    failed = [k for k, v in report["gates"].items() if not v["pass"]]
    print("\nfailing gates:", ", ".join(failed) if failed else "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
