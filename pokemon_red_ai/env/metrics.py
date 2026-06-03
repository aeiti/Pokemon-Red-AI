"""Per-step rubric metrics — pure read of the emulator's current state.

Single source of truth for the dense scalars listed under "Evaluation
rubric" in CLAUDE.md. Both the training logger and the eval harness
import this function so they cannot drift.

Cross-frame stats (unique maps visited, episode length) live in
:mod:`pokemon_red_ai.env.accumulator` so this function stays pure and
trivially testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pokemon_red_ai.emulator import ram_map

if TYPE_CHECKING:
    from pyboy import PyBoy

MetricsDict = dict[str, float]


def metrics(pyboy: "PyBoy") -> MetricsDict:
    """Snapshot scalars derived from the emulator's current RAM."""
    party_count = ram_map.read_party_count(pyboy)
    levels = ram_map.read_party_levels(pyboy)
    hp_fractions = ram_map.read_party_hp_fractions(pyboy)
    avg_hp = sum(hp_fractions) / party_count if party_count else 0.0

    return {
        "progress/badges": ram_map.read_badges_count(pyboy),
        "progress/boulder_badge": int(ram_map.has_boulder_badge(pyboy)),
        "progress/event_flags_set": ram_map.read_event_flags_set_count(pyboy),
        "party/count": party_count,
        "party/level_sum": sum(levels),
        "party/avg_hp_fraction": avg_hp,
        "economy/money": ram_map.read_money(pyboy),
        "pokedex/seen": ram_map.read_pokedex_seen_count(pyboy),
        "pokedex/caught": ram_map.read_pokedex_owned_count(pyboy),
    }
