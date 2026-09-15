"""Grader-private material: setting C, the official pool seed and the run seeds.

A solving agent never sees this file.  It is kept in the repository because the
task has to be auditable and reproducible by its reviewers, not because the agent
is meant to read it.
"""

from __future__ import annotations

from crossphase.core.generator import SettingSpec
from crossphase.core.settings import SETTING_A, SETTING_B, make_envs

#: Seed used for the official source and target pools.  The public diagnostic
#: uses a different one, so the agent cannot reconstruct the official pools
#: exactly even for the settings whose generator it holds.
OFFICIAL_POOL_SEED = 5501

#: Run seeds for the official measurement.  Three per setting.
OFFICIAL_RUN_SEEDS = (11, 12, 13, 14, 15)

#: The hidden setting.
#:
#: C changes the *mechanism* of the nuisance rather than its strength.  Instead of
#: adding a DC offset and a carrier, the level cue becomes a multiplicative gain
#: and the band cue becomes a monotone drift envelope.  Gain preserves the core's
#: shape -- and therefore the cross-channel phase relationship -- while changing
#: the scale of everything downstream of it, including the logits.  Objectives
#: keyed to logit magnitude or to risk *variance* therefore behave differently
#: here than they do under A and B's additive offsets.
#:
#: C also runs at a longer window with a weaker core and heavier observation
#: noise.  The nuisance-to-signal ratio is lower than in A and B but the noise
#: floor is higher, which is what makes a penalty strength selected on A and B
#: over-regularise here.
SETTING_C = SettingSpec(
    name="C",
    length=160,
    f_core=6.0,
    a_core=0.85,
    delta=0.20,
    mode="multiplicative",
    gamma=0.55,
    beta=0.90,
    envs=make_envs(
        levels=(0.790, 0.820, 0.850, 0.880, 0.805, 0.865),
        splits=(0.07, -0.05, 0.05, -0.07, 0.03, -0.03),
        # Noise rises with contamination here: the inverse of A and B.
        sigmas=(0.55, 0.62, 0.69, 0.76, 0.58, 0.72),
    ),
    source_sigma=0.66,
    source_mix=(0.85, 0.85),
)

ALL_SETTINGS = {"A": SETTING_A, "B": SETTING_B, "C": SETTING_C}
