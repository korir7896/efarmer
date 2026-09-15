"""Author-side invariants.

These reference the hidden setting, the official evaluation spec and the
packaging tool, so this module is deliberately absent from the agent bundle --
``tools/build_agent_bundle.py`` would flag it, which is how the split was found.
"""

from __future__ import annotations

from crossphase.core.engine import TRAIN, run_setting
from crossphase.core.methods import erm
from crossphase.core.settings import PROXY_WORLDS, SETTING_A, SETTING_B
from grader.private_specs import OFFICIAL_WORLDS, SETTING_C


def test_environment_count_distribution_is_setting_independent():
    for seed in range(50):
        counts = {s.env_count_for_seed(seed)
                  for s in (SETTING_A, SETTING_B, SETTING_C)}
        assert len(counts) == 1, "env count must not identify the setting"
        assert 4 <= counts.pop() <= 6


def test_worlds_are_four_tuples():
    for world in OFFICIAL_WORLDS + PROXY_WORLDS:
        assert len(world) == 4
        p1, p2, sigma_mult, core_mult = world
        assert 0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0
        assert sigma_mult > 0 and core_mult > 0


def test_official_and_proxy_worlds_are_disjoint():
    assert not (set(OFFICIAL_WORLDS) & set(PROXY_WORLDS))


def test_end_to_end_run_is_reproducible():
    cfg = TRAIN.__class__(steps=12)
    args = (SETTING_C, erm, 11, OFFICIAL_WORLDS, 300, 200, cfg)
    first = run_setting(*args, pool_seed=999)
    second = run_setting(*args, pool_seed=999)
    assert first["Q"] == second["Q"]


def test_agent_bundle_contains_no_private_material():
    """The packaged agent bundle must not carry the hidden setting, the official
    evaluation spec, the baseline scores or the reference solution."""
    from tools.build_agent_bundle import audit, build
    problems = audit(build())
    assert not problems, "\n".join(problems)


def test_public_settings_module_hides_the_official_spec():
    import crossphase.core.settings as public
    for name in ("OFFICIAL_WORLDS", "OFFICIAL_SOURCE_N", "OFFICIAL_WORLD_N",
                 "SETTING_C"):
        assert not hasattr(public, name), f"{name} is reachable from public code"

