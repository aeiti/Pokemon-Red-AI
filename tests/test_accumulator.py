"""EpisodeAccumulator tests — pure Python, no emulator boot.

The accumulator only reads ``pyboy.memory[CURRENT_MAP]``, so a tiny
stand-in with a dict-backed ``memory`` is sufficient. This keeps the
suite fast and lets us cover edge cases (revisits, resets, large IDs)
that would be a pain to engineer in the real game.
"""

from __future__ import annotations

import pytest

from pokemon_red_ai.emulator import ram_map
from pokemon_red_ai.env.accumulator import EpisodeAccumulator


class FakePyBoy:
    """Just enough of a PyBoy to satisfy the accumulator."""

    def __init__(self, map_id: int = 0) -> None:
        self.memory = {ram_map.CURRENT_MAP: map_id}

    def set_map(self, map_id: int) -> None:
        self.memory[ram_map.CURRENT_MAP] = map_id


def test_fresh_accumulator_snapshot_is_zero():
    acc = EpisodeAccumulator()
    snap = acc.snapshot()
    assert snap == {
        "progress/unique_maps_visited": 0,
        "progress/max_map_id_reached": 0,
        "episode/length_steps": 0,
    }


def test_update_counts_steps_and_tracks_visits():
    acc = EpisodeAccumulator()
    fake = FakePyBoy(map_id=0)
    acc.update(fake)
    snap = acc.snapshot()
    assert snap["episode/length_steps"] == 1
    assert snap["progress/unique_maps_visited"] == 1
    assert snap["progress/max_map_id_reached"] == 0


def test_revisited_map_does_not_double_count():
    acc = EpisodeAccumulator()
    fake = FakePyBoy(map_id=5)
    acc.update(fake)
    acc.update(fake)
    acc.update(fake)
    snap = acc.snapshot()
    assert snap["episode/length_steps"] == 3
    assert snap["progress/unique_maps_visited"] == 1
    assert snap["progress/max_map_id_reached"] == 5


def test_max_map_id_only_grows():
    acc = EpisodeAccumulator()
    fake = FakePyBoy()
    for m in [5, 10, 3, 7, 10, 1]:
        fake.set_map(m)
        acc.update(fake)
    snap = acc.snapshot()
    assert snap["progress/max_map_id_reached"] == 10
    assert snap["progress/unique_maps_visited"] == 5      # {5,10,3,7,1}
    assert snap["episode/length_steps"] == 6


def test_reset_clears_state():
    acc = EpisodeAccumulator()
    fake = FakePyBoy(map_id=42)
    for _ in range(5):
        acc.update(fake)
    acc.reset()
    assert acc.snapshot() == {
        "progress/unique_maps_visited": 0,
        "progress/max_map_id_reached": 0,
        "episode/length_steps": 0,
    }


def test_snapshot_is_non_destructive():
    acc = EpisodeAccumulator()
    fake = FakePyBoy(map_id=3)
    acc.update(fake)
    a = acc.snapshot()
    b = acc.snapshot()
    assert a == b


def test_read_only_views():
    acc = EpisodeAccumulator()
    fake = FakePyBoy(map_id=7)
    acc.update(fake)
    fake.set_map(11)
    acc.update(fake)
    assert acc.steps == 2
    assert acc.maps_visited == frozenset({7, 11})
    with pytest.raises((AttributeError, TypeError)):
        acc.maps_visited.add(99)  # frozenset is immutable
