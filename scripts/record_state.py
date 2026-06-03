"""Interactive save-state recorder.

Opens PyBoy in an SDL window so you can play with the keyboard. Hit F1
any time to write a save state to ``saved_states/``. Close the window
to quit.

PyBoy's default keymap:
  Arrow keys = D-pad
  A          = A button
  S          = B button
  Return     = Start
  Backspace  = Select
  Space      = turbo speed
  Esc        = quit

Output naming:
  - Default: timestamped ``snap_HHMMSS.state`` so you can save many
    candidates in one session and rename the keeper afterward.
  - ``--output NAME.state``: always write to that exact file
    (overwriting on each F1 press). Use this when you know the final
    name in advance.

Usage:
    uv run python scripts/record_state.py
    uv run python scripts/record_state.py --output v1_starter_chosen.state
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import sdl2

from pokemon_red_ai.emulator.pyboy_env import PyBoyEmulator

REPO_ROOT = Path(__file__).resolve().parents[1]
SAVED_STATES_DIR = REPO_ROOT / "saved_states"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Force a specific output filename (relative to saved_states/). "
             "If omitted, snapshots are timestamped.",
    )
    args = parser.parse_args()

    SAVED_STATES_DIR.mkdir(parents=True, exist_ok=True)

    print("Recording session started.")
    print("  F1: save state    |    Close window or Esc: quit")
    print("  Save confirmation appears in the terminal AND the window title bar.")
    if args.output:
        print(f"  Every F1 will overwrite saved_states/{args.output}")
    else:
        print("  Each F1 writes a new saved_states/snap_HHMMSS.state")
    print()

    save_count = 0
    prev_f1 = False

    with PyBoyEmulator(render=True) as emu:
        while emu.tick(1, render=True):
            # Non-consuming keyboard query — coexists with PyBoy's own
            # event handling for the Game Boy buttons.
            keys = sdl2.SDL_GetKeyboardState(None)
            f1_now = bool(keys[sdl2.SDL_SCANCODE_F1])
            if f1_now and not prev_f1:
                path = SAVED_STATES_DIR / (args.output or _timestamped_name())
                emu.save_state(path)
                save_count += 1
                print(f"  [saved #{save_count}] {path.relative_to(REPO_ROOT)}")
                emu.pyboy.title_status = f"saved #{save_count} → {path.name}"
            prev_f1 = f1_now

    print(f"\nSession ended. {save_count} state(s) saved.")


def _timestamped_name() -> str:
    return f"snap_{dt.datetime.now().strftime('%H%M%S')}.state"


if __name__ == "__main__":
    main()
