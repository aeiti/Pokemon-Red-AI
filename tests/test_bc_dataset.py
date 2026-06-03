"""DemoDataset tests — write synthetic HDF5 demos, load via the dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pokemon_red_ai.agent.bc_dataset import DemoDataset
from pokemon_red_ai.env.demo_format import save_demo
from pokemon_red_ai.env.observation import OBSERVATION_DIM


def _write_demo(path: Path, n: int, *, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    save_demo(
        path,
        observations=rng.random((n, OBSERVATION_DIM), dtype=np.float32),
        actions=rng.integers(0, 9, size=n, dtype=np.int8),
        rewards=rng.standard_normal(n).astype(np.float32),
        terminals=np.zeros(n, dtype=bool),
        rom_sha256="x",
        frame_skip=24,
        start_state="saved_states/v1_starter_chosen.state",
        recorded_at="2026-06-03T00:00:00",
        human_player=True,
    )


def test_single_demo_loads_with_correct_shapes(tmp_path):
    path = tmp_path / "one.h5"
    _write_demo(path, n=12)

    ds = DemoDataset([path])
    assert len(ds) == 12
    obs, act = ds[0]
    assert obs.shape == (OBSERVATION_DIM,)
    assert obs.dtype == torch.float32
    assert act.dtype == torch.int64
    assert 0 <= int(act) <= 8


def test_multiple_demos_are_concatenated(tmp_path):
    p1, p2 = tmp_path / "a.h5", tmp_path / "b.h5"
    _write_demo(p1, n=7, seed=1)
    _write_demo(p2, n=11, seed=2)

    ds = DemoDataset([p1, p2])
    assert len(ds) == 18
    assert ds.observations.shape == (18, OBSERVATION_DIM)
    assert ds.actions.shape == (18,)
    assert {Path(p) for p in ds.paths} == {p1, p2}


def test_empty_path_list_raises(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        DemoDataset([])


def test_observation_width_mismatch_raises(tmp_path):
    """A demo with wrong obs width must fail loudly, not silently truncate."""
    import h5py
    path = tmp_path / "wrong.h5"
    _write_demo(path, n=3)
    # Tamper: replace observations dataset with a wrong-width version.
    with h5py.File(path, "r+") as f:
        del f["observations"]
        f.create_dataset(
            "observations",
            data=np.zeros((3, OBSERVATION_DIM - 1), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="does not match"):
        DemoDataset([path])


def test_paths_are_normalized_to_path_objects(tmp_path):
    path = tmp_path / "p.h5"
    _write_demo(path, n=4)
    ds = DemoDataset([str(path)])               # passed as string
    assert all(isinstance(p, Path) for p in ds.paths)
