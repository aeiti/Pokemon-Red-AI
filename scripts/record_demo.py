"""Interactive demonstration recorder.

Plays Pokemon Red in an SDL window and logs (observation, action,
reward, terminal) at the env's decision cadence (every frame_skip
Game Boy frames). Close the window to stop; the trajectory is saved
to ``demonstrations/demo_<UTC-timestamp>.h5`` unless ``--output`` is
given.

Keys are PyBoy's defaults:
  arrows = D-pad
  A      = A button
  S      = B button
  Enter  = Start
  Backspace = Select
  Space  = turbo

Plus our additions:
  F1     = save an emulator save state right now (mid-session resume point)

On exit, the emulator's final state is auto-written to the demo's
companion .state file so the next session can resume from there:

  uv run python scripts/record_demo.py \\
      --start demonstrations/demo_<previous-timestamp>.state

What gets recorded per step:
  observations[i]  = obs vector right before step i
  actions[i]       = action decoded from the keys held at that moment
                     (priority: A > B > Start > Select > directions > no-op)
  rewards[i]       = compute_overworld_reward(prev_metrics, curr_metrics)
  terminals[i]     = True only on the last step of the session

Usage:
    uv run python scripts/record_demo.py
    uv run python scripts/record_demo.py --output demonstrations/run42.h5
    uv run python scripts/record_demo.py --start saved_states/my_state.state
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import sdl2

from pokemon_red_ai.emulator.pyboy_env import (
    PyBoyEmulator,
    disable_state_mutation_keys,
    resolve_rom_path,
)
from pokemon_red_ai.env.demo_format import save_demo, sha256_file
from pokemon_red_ai.env.keyboard_action import poll_pressed_scancodes, decode_action
from pokemon_red_ai.env.metrics import metrics
from pokemon_red_ai.env.observation import build_observation
from pokemon_red_ai.env.overworld_env import DEFAULT_FRAME_SKIP, DEFAULT_STATE
from pokemon_red_ai.env.reward import compute_overworld_reward

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMOS_DIR = REPO_ROOT / "demonstrations"


def _display(path: Path) -> str:
    """Show ``path`` relative to the repo when possible, else absolute."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output HDF5 path. Defaults to demonstrations/demo_<UTC-timestamp>.h5.",
    )
    parser.add_argument(
        "--start", type=Path, default=DEFAULT_STATE,
        help="Save state to load at the start of the session.",
    )
    parser.add_argument(
        "--frame-skip", type=int, default=DEFAULT_FRAME_SKIP,
        help="Game Boy frames per recorded step (default 24).",
    )
    args = parser.parse_args()

    if not args.start.is_file():
        raise SystemExit(f"Start state not found: {args.start}")

    disable_state_mutation_keys()

    output_path = args.output or DEMOS_DIR / _default_output_name()
    auto_state_path = output_path.with_suffix(".state")
    rom_path = resolve_rom_path()
    rom_hash = sha256_file(rom_path)

    print("Recording demo session.")
    print(f"  Start state: {_display(args.start)}")
    print(f"  Output:      {_display(output_path)}")
    print(f"  Auto-state:  {_display(auto_state_path)} (written on exit)")
    print(f"  Frame skip:  {args.frame_skip} ({60 / args.frame_skip:.1f} decisions/sec)")
    print("  PyBoy save/load/rewind keys (X, Z, comma, period) are disabled.")
    print("  F1: take a save-state snapshot mid-session.")
    print("  Close the window when done.")
    print()

    obs_buf: list[np.ndarray] = []
    act_buf: list[int] = []
    rew_buf: list[float] = []

    snap_count = 0
    prev_f1 = False

    with PyBoyEmulator(rom_path=rom_path, render=True) as emu:
        emu.load_state(args.start)
        emu.tick(1, render=True)
        prev_metrics = metrics(emu.pyboy)

        running = True
        while running:
            obs = build_observation(emu.pyboy)
            action = decode_action(poll_pressed_scancodes())

            # Let PyBoy run for frame_skip frames; the player's keys
            # drive the game directly via SDL2. Sample F1 every frame
            # so a brief tap can't be missed across the step window.
            for _ in range(args.frame_skip):
                if not emu.tick(1, render=True):
                    running = False
                    break
                keys = sdl2.SDL_GetKeyboardState(None)
                f1_now = bool(keys[sdl2.SDL_SCANCODE_F1])
                if f1_now and not prev_f1:
                    snap_count += 1
                    snap_path = output_path.parent / (
                        f"{output_path.stem}_snap_{snap_count}.state"
                    )
                    emu.save_state(snap_path)
                    print(f"  [F1 snapshot #{snap_count}] {_display(snap_path)}")
                prev_f1 = f1_now

            curr_metrics = metrics(emu.pyboy)
            reward = compute_overworld_reward(prev_metrics, curr_metrics)
            prev_metrics = curr_metrics

            obs_buf.append(obs)
            act_buf.append(action)
            rew_buf.append(reward)

        # Auto-save the final emulator state while PyBoy is still alive.
        emu.save_state(auto_state_path)

    n = len(act_buf)
    if n == 0:
        print("No steps recorded; nothing to save.")
        # The auto-state was still written — useful even with no steps.
        print(f"Final emulator state saved to {_display(auto_state_path)}.")
        return

    terminals = np.zeros(n, dtype=bool)
    terminals[-1] = True

    save_demo(
        output_path,
        observations=np.stack(obs_buf, axis=0),
        actions=np.asarray(act_buf, dtype=np.int8),
        rewards=np.asarray(rew_buf, dtype=np.float32),
        terminals=terminals,
        rom_sha256=rom_hash,
        frame_skip=args.frame_skip,
        start_state=_display(args.start),
        recorded_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        human_player=True,
    )

    print(f"\nSession ended. Saved {n} steps to {_display(output_path)}")
    print(f"Final emulator state at {_display(auto_state_path)} — "
          f"pass to --start next session to resume.")
    if snap_count:
        print(f"{snap_count} mid-session snapshot(s) also written.")


def _default_output_name() -> str:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"demo_{ts}.h5"


if __name__ == "__main__":
    main()
