"""Manual smoke check: load the ROM, run a few seconds, print observations.

Usage:
    uv run python scripts/smoke_emulator.py
    uv run python scripts/smoke_emulator.py --render        # watch it
    uv run python scripts/smoke_emulator.py --frames 1800   # 30s headless

Prints map ID, player position, party state, and badge/event-flag counts
roughly once per simulated second. If the values move when you'd expect
them to (e.g. coords change as you'd press buttons), the emulator and
RAM map are wired up correctly.
"""

from __future__ import annotations

import argparse

from pokemon_red_ai.emulator import ram_map
from pokemon_red_ai.emulator.pyboy_env import PyBoyEmulator

FRAMES_PER_SECOND = 60


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=600, help="Total frames to run")
    parser.add_argument(
        "--report-every", type=int, default=FRAMES_PER_SECOND,
        help="Print a snapshot every N frames (default 60 = ~1 second)",
    )
    parser.add_argument("--render", action="store_true", help="Open an SDL2 window")
    args = parser.parse_args()

    with PyBoyEmulator(render=args.render) as emu:
        for frame in range(0, args.frames, args.report_every):
            emu.tick(args.report_every, render=args.render)
            _print_snapshot(emu, frame + args.report_every)


def _print_snapshot(emu: PyBoyEmulator, frame: int) -> None:
    pyboy = emu.pyboy
    map_id, x, y = ram_map.read_position(pyboy)
    print(
        f"frame={frame:5d}  map={map_id:3d}  pos=({x:2d},{y:2d})  "
        f"party={ram_map.read_party_count(pyboy)}  "
        f"levels={ram_map.read_party_levels(pyboy)}  "
        f"badges={ram_map.read_badges_count(pyboy)}  "
        f"events={ram_map.read_event_flags_set_count(pyboy)}  "
        f"money=${ram_map.read_money(pyboy)}  "
        f"battle={ram_map.read_is_in_battle(pyboy)}"
    )


if __name__ == "__main__":
    main()
