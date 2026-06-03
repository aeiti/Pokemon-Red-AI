"""Print the in-game contents of a save state.

Loads a .state file in a headless emulator, ticks once so the RAM
reflects the loaded state, and prints the rubric metrics. Use this to
verify a freshly-recorded state contains what you think it does
before committing it to saved_states/.

Usage:
    uv run python scripts/inspect_state.py saved_states/v1_starter_chosen.state
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pokemon_red_ai.emulator import ram_map
from pokemon_red_ai.emulator.pyboy_env import PyBoyEmulator

# Pokemon Red species IDs for the starters (and Pikachu just in case).
# Source: pret/pokered/constants/pokemon_constants.asm
STARTER_SPECIES = {
    0x99: "Bulbasaur",
    0xB0: "Charmander",
    0xB1: "Squirtle",
    0x54: "Pikachu",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state_path", type=Path, help="Path to a .state file")
    args = parser.parse_args()

    if not args.state_path.is_file():
        raise SystemExit(f"State file not found: {args.state_path}")

    with PyBoyEmulator() as emu:
        emu.load_state(args.state_path)
        emu.tick(1)

        pyboy = emu.pyboy
        map_id, x, y = ram_map.read_position(pyboy)
        party_count = ram_map.read_party_count(pyboy)
        levels = ram_map.read_party_levels(pyboy)
        hp = ram_map.read_party_hp_fractions(pyboy)

        # Read raw species bytes from the party — first byte of each
        # party-mon struct is the species ID.
        species_ids = [
            pyboy.memory[ram_map.PARTY_MON_STRUCT + i * ram_map.PARTY_MON_STRIDE]
            for i in range(party_count)
        ]
        species_names = [
            STARTER_SPECIES.get(sid, f"#{sid:#04x}") for sid in species_ids
        ]

        print(f"State:    {args.state_path}")
        print(f"Map ID:   {map_id}")
        print(f"Position: ({x}, {y})")
        print(f"Party:    {party_count} mon(s)")
        for i, (name, lvl, frac) in enumerate(zip(species_names, levels, hp)):
            print(f"          [{i}] {name}  lvl {lvl}  hp {frac*100:.0f}%")
        print(f"Badges:   {ram_map.read_badges_count(pyboy)}")
        print(f"Money:    ${ram_map.read_money(pyboy)}")
        print(f"Pokedex:  seen {ram_map.read_pokedex_seen_count(pyboy)}, "
              f"caught {ram_map.read_pokedex_owned_count(pyboy)}")
        print(f"Events:   {ram_map.read_event_flags_set_count(pyboy)} flags set")
        print(f"Battle:   {ram_map.read_is_in_battle(pyboy)} "
              f"(0=overworld, 1=wild, 2=trainer)")


if __name__ == "__main__":
    main()
