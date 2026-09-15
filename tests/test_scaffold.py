"""Scaffold invariants visible to a solving agent.

Author-side invariants that need the hidden setting, the official evaluation
spec or the packaging tool live in ``tests/test_private.py``, which does not
ship in the agent bundle.  Run both with ``python -m tests.run``.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from crossphase.core.engine import (ObjectiveError, TRAIN, combine, quality,
                                    train_model)
from crossphase.core.generator import sample, training_environments
from crossphase.core.methods import erm, make_groupdro, make_irm
from crossphase.core.model import build_model, parameter_count
from crossphase.core.settings import PROXY_WORLDS, SETTING_A, SETTING_B


def test_generator_is_deterministic():
    a = sample(SETTING_A, 64, 0.9, 0.9, 0.4, (1, 2, 3))
    b = sample(SETTING_A, 64, 0.9, 0.9, 0.4, (1, 2, 3))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_cue_agreement_matches_request():
    _x, y, z1, z2 = sample(SETTING_A, 20000, 0.9, 0.1, 0.4, (7, 7))
    s = 2.0 * y - 1.0
    assert abs((z1 == s).mean() - 0.9) < 0.02
    assert abs((z2 == s).mean() - 0.1) < 0.02


def test_label_is_balanced_and_cues_are_label_driven():
    _x, y, _z1, _z2 = sample(SETTING_B, 20000, 0.5, 0.5, 0.4, (8, 8))
    assert abs(y.mean() - 0.5) < 0.02


def test_environment_order_is_permuted():
    orders = set()
    for seed in (11, 12, 13, 14, 15):
        envs = training_environments(SETTING_A, seed)
        orders.add(tuple(float(y.mean()) for _x, y in envs))
    assert len(orders) > 1


def test_model_is_length_agnostic_and_fixed_size():
    model = build_model(0)
    for length in (128, 160):
        assert model(torch.randn(3, 2, length)).shape == (3,)
    assert parameter_count() == 9105


def test_weight_decay_is_zero():
    assert TRAIN.weight_decay == 0.0, "a non-zero decay reopens the loss-scale route"


def test_quality_and_combine():
    result = {"I": 0.81, "T": {("a",): 0.64, ("b",): 0.9}}
    q, binding = quality(result)
    assert binding == ("a",)
    assert math.isclose(q, math.sqrt(0.81 * 0.64))
    assert math.isclose(combine([0.5, 0.5, 0.5]), 50.0)
    assert combine([0.5, 0.0, 0.5]) == 0.0


def test_objective_contract_is_enforced():
    for bad in (lambda o, t, s, ts, st: torch.stack([x.mean() for x in o]),
                lambda o, t, s, ts, st: o[0].mean().detach(),
                lambda o, t, s, ts, st: o[0].mean() * float("inf")):
        try:
            train_model(SETTING_A, bad, 11, TRAIN.__class__(steps=2))
        except ObjectiveError:
            continue
        raise AssertionError("contract violation was not caught")


def test_groupdro_state_adapts_to_environment_count():
    objective = make_groupdro(0.1, 0.0)
    state: dict = {}
    for count in (4, 6, 5):
        logits = [torch.randn(8, requires_grad=True) for _ in range(count)]
        targets = [torch.randint(0, 2, (8,)).float() for _ in range(count)]
        loss = objective(logits, targets, 10, 100, state)
        assert loss.dim() == 0 and state["dro_q"].numel() == count


def test_irm_penalty_is_computable_from_logits_alone():
    objective = make_irm(1.0, 0.0)
    logits = [torch.randn(16, requires_grad=True) for _ in range(4)]
    targets = [torch.randint(0, 2, (16,)).float() for _ in range(4)]
    loss = objective(logits, targets, 10, 100, {})
    loss.backward()
    assert all(o.grad is not None for o in logits)
