"""Observation builder tests, anchored to v1_starter_chosen.state."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pokemon_red_ai.env.observation import OBSERVATION_DIM, build_observation

V1_STATE = Path(__file__).resolve().parents[1] / "saved_states" / "v1_starter_chosen.state"


@pytest.fixture
def starter_emulator(emulator):
    if not V1_STATE.is_file():
        pytest.skip(f"v1 starter state missing at {V1_STATE}")
    emulator.load_state(V1_STATE)
    emulator.tick(1)
    return emulator


def test_observation_shape_and_dtype(starter_emulator):
    obs = build_observation(starter_emulator.pyboy)
    assert obs.shape == (OBSERVATION_DIM,)
    assert obs.dtype == np.float32


def test_observation_values_are_in_unit_range(starter_emulator):
    obs = build_observation(starter_emulator.pyboy)
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)
    assert np.all(np.isfinite(obs))


def test_observation_v1_starter_anchors(starter_emulator):
    """Specific dims have known values at the v1 starter state.

    If this fails, either the obs schema changed or the save state
    was overwritten. Either case wants a human in the loop.
    """
    obs = build_observation(starter_emulator.pyboy)

    # Map 0 (Pallet Town), position (5, 6)
    assert obs[0] == pytest.approx(0.0)
    assert obs[1] == pytest.approx(5 / 255.0)
    assert obs[2] == pytest.approx(6 / 255.0)
    # 1 mon in party
    assert obs[3] == pytest.approx(1 / 6.0)
    # First party slot = Charmander (species 0xB0)
    assert obs[4] == pytest.approx(0xB0 / 255.0)
    # First party level = 6
    assert obs[10] == pytest.approx(6 / 100.0)
    # First party HP fraction = 1.0
    assert obs[16] == pytest.approx(1.0)
    # No badges
    assert obs[22] == pytest.approx(0.0)
    assert obs[23] == pytest.approx(0.0)
    # Overworld, not in battle
    assert obs[28] == pytest.approx(1.0)
    assert obs[29] == pytest.approx(0.0)


def test_unused_party_slots_are_zero(starter_emulator):
    """Slots 1-5 are unfilled with one party member."""
    obs = build_observation(starter_emulator.pyboy)
    for i in range(1, 6):                # species slots
        assert obs[4 + i] == pytest.approx(0.0)
    for i in range(1, 6):                # level slots
        assert obs[10 + i] == pytest.approx(0.0)
    for i in range(1, 6):                # hp slots
        assert obs[16 + i] == pytest.approx(0.0)


def test_observation_is_pure(starter_emulator):
    a = build_observation(starter_emulator.pyboy)
    b = build_observation(starter_emulator.pyboy)
    assert np.array_equal(a, b)
