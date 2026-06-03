"""Per-episode running statistics that need cross-frame state.

The env owns one of these per episode. Call :meth:`update` once per
step with the live PyBoy instance, then read :meth:`snapshot` whenever
you want the current accumulated values (e.g. at episode end for the
final scalars, or every N steps for intermediate logging).

This is deliberately separate from :mod:`pokemon_red_ai.env.metrics`
so that file can stay a pure snapshot function. The two combine in the
env loop:

    accumulator.update(pyboy)
    log = {**metrics(pyboy), **accumulator.snapshot()}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pokemon_red_ai.emulator import ram_map

if TYPE_CHECKING:
    from pyboy import PyBoy


class EpisodeAccumulator:
    """Track exploration and length across one episode."""

    def __init__(self) -> None:
        self._steps: int = 0
        self._maps_visited: set[int] = set()
        self._max_map_id: int = 0

    def reset(self) -> None:
        """Clear all state. Call once at the start of each episode."""
        self._steps = 0
        self._maps_visited = set()
        self._max_map_id = 0

    def update(self, pyboy: "PyBoy") -> None:
        """Advance one step's worth of bookkeeping."""
        self._steps += 1
        map_id = int(pyboy.memory[ram_map.CURRENT_MAP])
        self._maps_visited.add(map_id)
        if map_id > self._max_map_id:
            self._max_map_id = map_id

    def snapshot(self) -> dict[str, float]:
        return {
            "progress/unique_maps_visited": len(self._maps_visited),
            "progress/max_map_id_reached": self._max_map_id,
            "episode/length_steps": self._steps,
        }

    # --- read-only views for env-internal logic ---------------------------

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def maps_visited(self) -> frozenset[int]:
        return frozenset(self._maps_visited)
