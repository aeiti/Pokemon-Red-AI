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
    rom_path = resolve_rom_path()
    rom_hash = sha256_file(rom_path)

    print("Recording demo session.")
    print(f"  Start state: {args.start.relative_to(REPO_ROOT)}")
    print(f"  Output:      {output_path.relative_to(REPO_ROOT)}")
    print(f"  Frame skip:  {args.frame_skip} ({60 / args.frame_skip:.1f} decisions/sec)")
    print("  PyBoy save/load/rewind keys (X, Z, comma, period) are disabled.")
    print("  Play normally. Close the window when done.")
    print()

    obs_buf: list[np.ndarray] = []
    act_buf: list[int] = []
    rew_buf: list[float] = []

    with PyBoyEmulator(rom_path=rom_path, render=True) as emu:
        emu.load_state(args.start)
        emu.tick(1, render=True)
        prev_metrics = metrics(emu.pyboy)

        running = True
        while running:
            obs = build_observation(emu.pyboy)
            action = decode_action(poll_pressed_scancodes())

            # Let PyBoy run for frame_skip frames; the player's keys
            # drive the game directly via SDL2.
            for _ in range(args.frame_skip):
                if not emu.tick(1, render=True):
                    running = False
                    break

            curr_metrics = metrics(emu.pyboy)
            reward = compute_overworld_reward(prev_metrics, curr_metrics)
            prev_metrics = curr_metrics

            obs_buf.append(obs)
            act_buf.append(action)
            rew_buf.append(reward)

    n = len(act_buf)
    if n == 0:
        print("No steps recorded; nothing to save.")
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
        start_state=str(args.start.relative_to(REPO_ROOT)),
        recorded_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        human_player=True,
    )

    print(f"\nSession ended. Saved {n} steps to {output_path}")


def _default_output_name() -> str:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"demo_{ts}.h5"


if __name__ == "__main__":
    main()
