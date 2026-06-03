"""HDF5 demonstration format — schema constants, save/load helpers.

One file = one play session. Datasets are aligned arrays of length N,
where N is the number of recorded steps:

    /observations  (N, 30)   float32     the env's observation vector
    /actions       (N,)      int8        0..8, the Discrete(9) action
    /rewards       (N,)      float32     reward at each step
    /terminals     (N,)      bool        episode-boundary marker

File-level attributes:

    format_version    str    bumped on any breaking schema change
    rom_sha256        str    hash of the ROM used to record
    frame_skip        int    Game Boy frames per recorded step
    start_state       str    path to the save state used as reset point
    recorded_at       str    ISO 8601 timestamp
    human_player      bool   True for human play; False for agent rollouts

The reward and terminal channels are recorded even though BC only
needs (obs, action). Keeping them future-proofs the format for offline
RL and DAgger without re-recording.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

FORMAT_VERSION = "1"

OBS_DATASET = "observations"
ACTIONS_DATASET = "actions"
REWARDS_DATASET = "rewards"
TERMINALS_DATASET = "terminals"

REQUIRED_ATTRS = (
    "format_version",
    "rom_sha256",
    "frame_skip",
    "start_state",
    "recorded_at",
    "human_player",
)


@dataclass
class Demo:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    format_version: str
    rom_sha256: str
    frame_skip: int
    start_state: str
    recorded_at: str
    human_player: bool

    def __len__(self) -> int:
        return len(self.actions)


def save_demo(
    path: str | Path,
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminals: np.ndarray,
    rom_sha256: str,
    frame_skip: int,
    start_state: str,
    recorded_at: str,
    human_player: bool = True,
) -> None:
    """Write a demo file. Caller is responsible for array dtypes/shapes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(actions)
    if not (len(observations) == len(rewards) == len(terminals) == n):
        raise ValueError(
            f"All arrays must have the same length; got "
            f"obs={len(observations)} actions={n} "
            f"rewards={len(rewards)} terminals={len(terminals)}"
        )

    with h5py.File(path, "w") as f:
        f.create_dataset(OBS_DATASET, data=observations.astype(np.float32), compression="gzip")
        f.create_dataset(ACTIONS_DATASET, data=actions.astype(np.int8), compression="gzip")
        f.create_dataset(REWARDS_DATASET, data=rewards.astype(np.float32), compression="gzip")
        f.create_dataset(TERMINALS_DATASET, data=terminals.astype(bool), compression="gzip")
        f.attrs["format_version"] = FORMAT_VERSION
        f.attrs["rom_sha256"] = rom_sha256
        f.attrs["frame_skip"] = int(frame_skip)
        f.attrs["start_state"] = start_state
        f.attrs["recorded_at"] = recorded_at
        f.attrs["human_player"] = bool(human_player)


def load_demo(path: str | Path) -> Demo:
    """Read a demo file. Raises ValueError on schema mismatch."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        for attr in REQUIRED_ATTRS:
            if attr not in f.attrs:
                raise ValueError(f"{path}: missing required attribute '{attr}'")

        version = _attr_str(f.attrs["format_version"])
        if version != FORMAT_VERSION:
            raise ValueError(
                f"{path}: unsupported format_version {version!r} "
                f"(expected {FORMAT_VERSION!r})"
            )

        return Demo(
            observations=f[OBS_DATASET][:],
            actions=f[ACTIONS_DATASET][:],
            rewards=f[REWARDS_DATASET][:],
            terminals=f[TERMINALS_DATASET][:],
            format_version=version,
            rom_sha256=_attr_str(f.attrs["rom_sha256"]),
            frame_skip=int(f.attrs["frame_skip"]),
            start_state=_attr_str(f.attrs["start_state"]),
            recorded_at=_attr_str(f.attrs["recorded_at"]),
            human_player=bool(f.attrs["human_player"]),
        )


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file, as a lowercase hex string. Used to tag the ROM."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _attr_str(value: object) -> str:
    """h5py returns bytes for some string attrs depending on write path."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
