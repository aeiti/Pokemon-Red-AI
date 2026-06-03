"""Print a summary of a recorded demonstration file.

Loads an HDF5 demo, prints metadata + per-channel stats so you can
sanity-check what you just recorded before training on it.

Usage:
    uv run python scripts/inspect_demo.py demonstrations/demo_20260603T....h5
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from pokemon_red_ai.env.demo_format import load_demo
from pokemon_red_ai.env.overworld_env import ACTIONS

ACTION_LABELS = tuple(b if b is not None else "no-op" for b in ACTIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to a demo .h5 file")
    args = parser.parse_args()

    if not args.path.is_file():
        raise SystemExit(f"Demo not found: {args.path}")

    demo = load_demo(args.path)
    n = len(demo)
    duration_sec = n * demo.frame_skip / 60.0

    print(f"File:         {args.path}")
    print(f"Format:       v{demo.format_version}")
    print(f"Recorded:     {demo.recorded_at}")
    print(f"Start state:  {demo.start_state}")
    print(f"ROM SHA-256:  {demo.rom_sha256[:16]}...")
    print(f"Human player: {demo.human_player}")
    print()
    print(f"Steps:        {n}")
    print(f"Frame skip:   {demo.frame_skip}")
    print(f"Duration:     ~{duration_sec:.1f}s of game time "
          f"({duration_sec / 60:.1f} min)")
    print()

    print("Action histogram:")
    counts = Counter(int(a) for a in demo.actions)
    for action_idx in range(len(ACTION_LABELS)):
        c = counts.get(action_idx, 0)
        pct = 100.0 * c / n if n else 0.0
        bar = "#" * int(pct / 2)
        print(f"  {ACTION_LABELS[action_idx]:<8} {c:>6} ({pct:5.1f}%)  {bar}")
    print()

    print("Reward:")
    rewards = demo.rewards
    print(f"  total:   {rewards.sum():.3f}")
    print(f"  nonzero: {int(np.count_nonzero(rewards))} steps")
    print(f"  max:     {rewards.max():.3f}")
    print(f"  min:     {rewards.min():.3f}")
    print()

    _summarize_progression(demo.observations, demo.frame_skip)


def _summarize_progression(obs: np.ndarray, frame_skip: int) -> None:
    """Show how key in-game scalars changed over the session.

    Reads them straight out of the observation vector by index (the
    schema is documented in observation.py). If observation.py's
    layout ever drifts, this output goes weird and we know to update.
    """
    map_id = (obs[:, 0] * 255).round().astype(int)
    badges = (obs[:, 22] * 8).round().astype(int)
    first_level = (obs[:, 10] * 100).round().astype(int)

    print("Progression:")
    print(f"  Map IDs visited:  {sorted(set(map_id.tolist()))}")
    print(f"  Badges:           start={badges[0]}, end={badges[-1]}, max={badges.max()}")
    print(f"  Lead-mon level:   start={first_level[0]}, end={first_level[-1]}, "
          f"max={first_level.max()}")


if __name__ == "__main__":
    main()
