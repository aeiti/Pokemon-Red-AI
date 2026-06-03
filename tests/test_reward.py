"""Reward function tests — pure, no emulator."""

from __future__ import annotations

from pokemon_red_ai.env.reward import compute_overworld_reward


def _m(level_sum: int = 0, boulder: int = 0) -> dict[str, float]:
    return {"party/level_sum": level_sum, "progress/boulder_badge": boulder}


def test_zero_when_state_unchanged():
    s = _m(level_sum=12, boulder=0)
    assert compute_overworld_reward(s, s) == 0.0


def test_level_up_yields_small_positive():
    prev = _m(level_sum=6)
    curr = _m(level_sum=7)
    assert compute_overworld_reward(prev, curr) == 0.01


def test_multi_level_delta_scales_linearly():
    prev = _m(level_sum=10)
    curr = _m(level_sum=15)
    assert compute_overworld_reward(prev, curr) == 0.05


def test_boulder_badge_dominates():
    prev = _m(level_sum=10, boulder=0)
    curr = _m(level_sum=10, boulder=1)
    assert compute_overworld_reward(prev, curr) == 100.0


def test_badge_and_level_combine_additively():
    prev = _m(level_sum=10, boulder=0)
    curr = _m(level_sum=12, boulder=1)
    assert compute_overworld_reward(prev, curr) == 100.02


def test_negative_delta_is_penalized():
    """If level_sum drops (e.g., a mon got released), reward should reflect that."""
    prev = _m(level_sum=10)
    curr = _m(level_sum=8)
    assert compute_overworld_reward(prev, curr) == -0.02
