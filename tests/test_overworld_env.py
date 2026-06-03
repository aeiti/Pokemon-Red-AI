"""Gymnasium conformance + smoke tests for PokemonRedOverworldEnv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pokemon_red_ai.env.observation import OBSERVATION_DIM
from pokemon_red_ai.env.overworld_env import (
    DEFAULT_STATE,
    NUM_ACTIONS,
    PokemonRedOverworldEnv,
)

if not DEFAULT_STATE.is_file():
    pytest.skip(
        f"v1 starter state missing at {DEFAULT_STATE}",
        allow_module_level=True,
    )


@pytest.fixture
def env(rom_path):
    """Small-budget env so tests stay fast. Auto-close."""
    env = PokemonRedOverworldEnv(
        rom_path=rom_path,
        max_steps=8,
        frame_skip=8,
    )
    try:
        yield env
    finally:
        env.close()


def test_spaces_match_contract(env):
    assert env.action_space.n == NUM_ACTIONS == 9
    assert env.observation_space.shape == (OBSERVATION_DIM,)
    assert env.observation_space.dtype == np.float32


def test_reset_returns_obs_and_info(env):
    obs, info = env.reset()
    assert obs.shape == (OBSERVATION_DIM,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    # info merges metrics + accumulator snapshot
    assert "progress/badges" in info
    assert "episode/length_steps" in info
    assert info["episode/length_steps"] == 0


def test_reset_is_deterministic(env):
    obs_a, _ = env.reset()
    obs_b, _ = env.reset()
    assert np.array_equal(obs_a, obs_b)


def test_step_returns_five_tuple(env):
    env.reset()
    result = env.step(8)                    # no-op
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "episode/length_steps" in info


def test_step_advances_accumulator(env):
    env.reset()
    _, _, _, _, info_after_one = env.step(8)
    assert info_after_one["episode/length_steps"] == 1
    _, _, _, _, info_after_two = env.step(8)
    assert info_after_two["episode/length_steps"] == 2


def test_truncation_fires_at_step_cap(env):
    env.reset()
    truncated = False
    for _ in range(env.max_steps):
        _, _, _, truncated, _ = env.step(8)
        if truncated:
            break
    assert truncated, "Expected truncated=True once max_steps was reached"


def test_step_before_reset_raises(rom_path):
    env = PokemonRedOverworldEnv(rom_path=rom_path, max_steps=4, frame_skip=4)
    try:
        with pytest.raises(RuntimeError):
            env.step(8)
    finally:
        env.close()


def test_gymnasium_env_checker(env):
    """The gymnasium-supplied checker catches contract violations."""
    from gymnasium.utils.env_checker import check_env

    # skip_render_check=True because we initialized with render_mode=None
    check_env(env, skip_render_check=True)
