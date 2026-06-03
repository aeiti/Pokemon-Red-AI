"""Tests for the metrics snapshot, anchored to the v1 starter state.

These tests double as a regression check on the committed save state:
if v1_starter_chosen.state ever drifts (re-recorded with different
in-game choices), this test will fail and force us to notice.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pokemon_red_ai.env.metrics import metrics

V1_STATE = Path(__file__).resolve().parents[1] / "saved_states" / "v1_starter_chosen.state"


@pytest.fixture
def starter_emulator(emulator):
    """An emulator with the v1 starter state loaded and ticked once."""
    if not V1_STATE.is_file():
        pytest.skip(f"v1 starter state missing at {V1_STATE}")
    emulator.load_state(V1_STATE)
    emulator.tick(1)
    return emulator


def test_metrics_returns_expected_keys(starter_emulator):
    """The dict has exactly the keys the rubric promises — no drift."""
    snapshot = metrics(starter_emulator.pyboy)
    expected = {
        "progress/badges",
        "progress/boulder_badge",
        "progress/event_flags_set",
        "party/count",
        "party/level_sum",
        "party/avg_hp_fraction",
        "economy/money",
        "pokedex/seen",
        "pokedex/caught",
    }
    assert set(snapshot.keys()) == expected


def test_metrics_v1_starter_values(starter_emulator):
    """Anchor: exact values for v1_starter_chosen.state.

    If you intentionally re-record the v1 state, update these numbers.
    Otherwise a mismatch means either (a) the state was overwritten
    accidentally or (b) a metric helper started returning wrong values.
    """
    snapshot = metrics(starter_emulator.pyboy)

    assert snapshot["progress/badges"] == 0
    assert snapshot["progress/boulder_badge"] == 0
    assert snapshot["progress/event_flags_set"] == 6

    assert snapshot["party/count"] == 1
    assert snapshot["party/level_sum"] == 6              # Charmander, lvl 6
    assert snapshot["party/avg_hp_fraction"] == pytest.approx(1.0)

    assert snapshot["economy/money"] == 3175
    assert snapshot["pokedex/seen"] == 2
    assert snapshot["pokedex/caught"] == 1


def test_metrics_handles_empty_party(emulator):
    """Title screen — no party loaded — avg HP must not divide by zero."""
    emulator.tick(60)
    snapshot = metrics(emulator.pyboy)
    assert snapshot["party/count"] == 0
    assert snapshot["party/level_sum"] == 0
    assert snapshot["party/avg_hp_fraction"] == 0.0


def test_metrics_is_pure(starter_emulator):
    """Two calls in a row return identical dicts — no hidden state."""
    a = metrics(starter_emulator.pyboy)
    b = metrics(starter_emulator.pyboy)
    assert a == b
