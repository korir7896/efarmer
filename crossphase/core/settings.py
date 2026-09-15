"""Public setting registry (A and B) plus the public proxy evaluation pools.

Setting C is not defined here.  It lives with the grader and is not visible to a
solving agent.

Environment parameterisation
----------------------------
Each environment is written as a *contamination level* ``l`` and a *cue split*
``s``: ``q1 = l + s`` and ``q2 = l - s``.  The split alternates in sign across
environments, so the two cues disagree about which environment favours them --
that is the variation an invariance penalty can actually use.  The level moves
monotonically against the noise scale, and the sign of that relationship is a
design knob:

* Settings A and B: level *decreases* as sigma increases, so the hardest
  environment is also the least shortcut-contaminated one.  Emphasising the
  worst environment is then aligned with suppressing the shortcut.
* Setting C (hidden): level *increases* as sigma increases, so the hardest
  environment is hard for reasons unrelated to the shortcut and is simultaneously
  the most contaminated.  Worst-environment emphasis is actively counter-productive
  there.  This is the mechanism behind the required rank reversal.
"""

from __future__ import annotations

from .generator import Environment, SettingSpec

#: Official target worlds, as (p1, p2) cue-agreement pairs.  The coordinates are
#: public -- the scoring rule is not a guessing game -- but the official pools are
#: drawn with a private seed, the official run seeds are private, and setting C's
#: generator is withheld entirely.
#: Three of them flip or split the cues, which punishes shortcut reliance; two
#: stress the core detector instead, which punishes over-regularisation.  Both
#: kinds are needed for the minimum to be contested -- see the binding-world audit
#: in ``reports/``.
#: Chosen from a ten-world profile across all three settings, not by assumption.
#: A harder cue-flip at (.05, .05) was measured and dropped: it sits uniformly
#: below (.10, .10), so it would bind everywhere and leave the rest decorative.
OFFICIAL_WORLDS = (
    (0.10, 0.10, 1.00, 1.00),   # both cues flipped
    (0.10, 0.90, 1.00, 1.00),   # cue conflict
    (0.90, 0.10, 1.00, 1.00),   # cue conflict, the other way
    (0.50, 0.50, 1.50, 1.00),   # cues uninformative, noise floor raised 50%
    (0.90, 0.90, 1.00, 0.45),   # cues aligned, core attenuated to 45%
)

#: Proxy worlds available to the agent.  Deliberately *not* the official
#: coordinates: milder disagreement, four worlds rather than five, and an
#: independent seed, so the world that binds the minimum under the proxy need not
#: be the world that binds it officially.
#: Moved closer to the official coordinates than the first draft, which sat at
#: (.20,.20) and correlated with the official score at only Spearman 0.25 --
#: below the 0.5 floor, where the twelve evaluations stop being feedback and
#: become noise.  Still deliberately distinct: different coordinates, four
#: worlds rather than five, an independent seed and smaller pools, so the world
#: that binds under the proxy need not be the one that binds officially.
PROXY_WORLDS = (
    (0.15, 0.15, 1.00, 1.00),
    (0.15, 0.85, 1.00, 1.00),
    (0.85, 0.15, 1.00, 1.00),
    (0.60, 0.60, 1.40, 0.55),
)

OFFICIAL_SOURCE_N = 2000
OFFICIAL_WORLD_N = 1200
PROXY_SOURCE_N = 900
PROXY_WORLD_N = 700

#: Run seeds used by the public diagnostic.  The grader uses different ones.
PUBLIC_RUN_SEEDS = (101, 102, 103)

#: Seed for the public proxy pools.  The official pool seed is grader-private.
PUBLIC_POOL_SEED = 202


def make_envs(levels, splits, sigmas) -> tuple:
    return tuple(
        Environment(round(l + s, 4), round(l - s, 4), sig)
        for l, s, sig in zip(levels, splits, sigmas)
    )


SPLITS = (0.07, -0.05, 0.05, -0.07, 0.03, -0.03)

#: Contamination levels sit near 0.75-0.85 rather than near 0.9.  With label
#: noise present, the shortcut only has to beat ``1 - label_noise`` to be worth
#: taking, and pushing the levels higher makes the escape from it bimodal --
#: measured, and the reason these numbers are where they are.

SETTING_A = SettingSpec(
    name="A",
    length=128,
    f_core=5.0,
    a_core=1.00,
    delta=0.45,
    mode="additive",
    label_noise=0.15,
    kappa1=1.40,
    kappa2=1.10,
    f_band=(12.0, 18.0),
    envs=make_envs(
        levels=(0.830, 0.800, 0.770, 0.740, 0.815, 0.755),
        splits=SPLITS,
        sigmas=(0.40, 0.44, 0.48, 0.52, 0.42, 0.50),
    ),
    source_sigma=0.46,
    source_mix=(0.80, 0.80),
)

SETTING_B = SettingSpec(
    name="B",
    length=128,
    f_core=7.0,
    a_core=1.05,
    delta=0.42,
    mode="additive",
    label_noise=0.18,
    kappa1=0.95,
    kappa2=1.55,
    f_band=(14.0, 22.0),
    envs=make_envs(
        levels=(0.845, 0.815, 0.785, 0.755, 0.830, 0.770),
        splits=tuple(-s for s in SPLITS),
        sigmas=(0.44, 0.49, 0.54, 0.59, 0.46, 0.57),
    ),
    source_sigma=0.51,
    source_mix=(0.81, 0.81),
)

PUBLIC_SETTINGS = {"A": SETTING_A, "B": SETTING_B}
