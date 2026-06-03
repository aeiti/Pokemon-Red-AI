"""PyTorch Dataset over one or many HDF5 demonstrations.

Loads each demo with ``demo_format.load_demo`` (which validates the
schema and rejects format mismatches), concatenates the observations
and actions, and serves them as ``(obs_tensor, action_tensor)``
indexable pairs. Stored as tensors up front so per-batch iteration is
zero-copy.

Rewards and terminals are present in the file but BC ignores them.
The training script preserves the file path list in its checkpoint
metadata so we can reproduce a run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from pokemon_red_ai.env.demo_format import load_demo
from pokemon_red_ai.env.observation import OBSERVATION_DIM


class DemoDataset(Dataset):
    """Concatenated (observation, action) pairs from one or many demos."""

    def __init__(self, paths: Iterable[Path | str]) -> None:
        paths = [Path(p) for p in paths]
        if not paths:
            raise ValueError("DemoDataset requires at least one demo path.")

        obs_arrays: list[np.ndarray] = []
        act_arrays: list[np.ndarray] = []
        for p in paths:
            demo = load_demo(p)
            if demo.observations.shape[1] != OBSERVATION_DIM:
                raise ValueError(
                    f"{p}: observation width {demo.observations.shape[1]} "
                    f"does not match current OBSERVATION_DIM={OBSERVATION_DIM}"
                )
            obs_arrays.append(demo.observations)
            act_arrays.append(demo.actions)

        obs = np.concatenate(obs_arrays, axis=0).astype(np.float32)
        actions = np.concatenate(act_arrays, axis=0).astype(np.int64)

        self._observations = torch.from_numpy(obs)
        self._actions = torch.from_numpy(actions)
        self._paths = paths

    def __len__(self) -> int:
        return self._actions.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._observations[idx], self._actions[idx]

    @property
    def observations(self) -> torch.Tensor:
        return self._observations

    @property
    def actions(self) -> torch.Tensor:
        return self._actions

    @property
    def paths(self) -> list[Path]:
        return list(self._paths)
