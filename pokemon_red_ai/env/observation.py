"""Build a normalized observation vector from the emulator's RAM.

Used by the overworld env (and, later, the battle env's overworld
context view). Output is a 1-D np.float32 vector with every entry in
[0.0, 1.0] so the policy network sees inputs of comparable scale and
the Gym Box space matches what it advertises.

Schema (30 dims, in order):
    [0]      map_id / 255
    [1]      player_x / 255
    [2]      player_y / 255
    [3]      party_count / 6
    [4-9]    party species ids / 255    (0 in unused slots)
    [10-15]  party levels / 100         (0 in unused slots)
    [16-21]  party HP fractions         (0 in unused slots)
    [22]     badges popcount / 8
    [23]     boulder_badge bit          (0 or 1)
    [24]     min(event_flags / 100, 1)
    [25]     pokedex_seen / 151
    [26]     pokedex_caught / 151
    [27]     money / 999999
    [28]     1 if overworld else 0
    [29]     1 if in any battle else 0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pokemon_red_ai.emulator import ram_map

if TYPE_CHECKING:
    from pyboy import PyBoy

OBSERVATION_DIM = 30


def build_observation(pyboy: "PyBoy") -> np.ndarray:
    """Pure read of current RAM into the 30-dim observation vector."""
    map_id, x, y = ram_map.read_position(pyboy)
    party_count = ram_map.read_party_count(pyboy)
    levels = ram_map.read_party_levels(pyboy)
    hp_fracs = ram_map.read_party_hp_fractions(pyboy)
    species = [
        pyboy.memory[ram_map.PARTY_MON_STRUCT + i * ram_map.PARTY_MON_STRIDE]
        for i in range(party_count)
    ]

    # Pad party slots to 6.
    species = species + [0] * (6 - len(species))
    levels = levels + [0] * (6 - len(levels))
    hp_fracs = hp_fracs + [0.0] * (6 - len(hp_fracs))

    is_battle = ram_map.read_is_in_battle(pyboy)
    event_flags_norm = ram_map.read_event_flags_set_count(pyboy) / 100.0

    obs = np.array(
        [
            map_id / 255.0,
            x / 255.0,
            y / 255.0,
            party_count / 6.0,
            *[s / 255.0 for s in species],
            *[lvl / 100.0 for lvl in levels],
            *hp_fracs,
            ram_map.read_badges_count(pyboy) / 8.0,
            float(ram_map.has_boulder_badge(pyboy)),
            min(event_flags_norm, 1.0),
            ram_map.read_pokedex_seen_count(pyboy) / 151.0,
            ram_map.read_pokedex_owned_count(pyboy) / 151.0,
            ram_map.read_money(pyboy) / 999_999.0,
            float(is_battle == 0),
            float(is_battle in (1, 2)),
        ],
        dtype=np.float32,
    )
    return obs
