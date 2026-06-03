"""Demo format roundtrip + schema-validation tests."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from pokemon_red_ai.env.demo_format import (
    FORMAT_VERSION,
    Demo,
    REQUIRED_ATTRS,
    load_demo,
    save_demo,
    sha256_file,
)


def _make_dummy(n: int = 5):
    rng = np.random.default_rng(0)
    obs = rng.random((n, 30), dtype=np.float32)
    actions = rng.integers(0, 9, size=n, dtype=np.int8)
    rewards = rng.standard_normal(n).astype(np.float32)
    terminals = np.zeros(n, dtype=bool)
    terminals[-1] = True
    return obs, actions, rewards, terminals


def test_save_load_roundtrip(tmp_path):
    obs, actions, rewards, terminals = _make_dummy()
    out = tmp_path / "demo.h5"
    save_demo(
        out,
        observations=obs, actions=actions, rewards=rewards, terminals=terminals,
        rom_sha256="abc123",
        frame_skip=24,
        start_state="saved_states/v1_starter_chosen.state",
        recorded_at="2026-06-03T00:00:00",
        human_player=True,
    )

    demo = load_demo(out)
    assert isinstance(demo, Demo)
    assert np.array_equal(demo.observations, obs)
    assert np.array_equal(demo.actions, actions)
    assert np.allclose(demo.rewards, rewards)
    assert np.array_equal(demo.terminals, terminals)
    assert demo.format_version == FORMAT_VERSION
    assert demo.rom_sha256 == "abc123"
    assert demo.frame_skip == 24
    assert demo.start_state == "saved_states/v1_starter_chosen.state"
    assert demo.recorded_at == "2026-06-03T00:00:00"
    assert demo.human_player is True
    assert len(demo) == 5


def test_save_creates_parent_directory(tmp_path):
    obs, actions, rewards, terminals = _make_dummy()
    out = tmp_path / "deeper" / "nested" / "demo.h5"
    save_demo(
        out,
        observations=obs, actions=actions, rewards=rewards, terminals=terminals,
        rom_sha256="x", frame_skip=24, start_state="x",
        recorded_at="x", human_player=True,
    )
    assert out.is_file()


def test_mismatched_lengths_raise(tmp_path):
    obs = np.zeros((3, 30), dtype=np.float32)
    actions = np.zeros(5, dtype=np.int8)
    rewards = np.zeros(5, dtype=np.float32)
    terminals = np.zeros(5, dtype=bool)
    with pytest.raises(ValueError, match="same length"):
        save_demo(
            tmp_path / "demo.h5",
            observations=obs, actions=actions, rewards=rewards, terminals=terminals,
            rom_sha256="x", frame_skip=24, start_state="x",
            recorded_at="x", human_player=True,
        )


def test_load_rejects_missing_attr(tmp_path):
    """A file without the required attrs should fail loudly, not silently."""
    out = tmp_path / "broken.h5"
    with h5py.File(out, "w") as f:
        f.create_dataset("observations", data=np.zeros((1, 30), dtype=np.float32))
        f.create_dataset("actions", data=np.zeros(1, dtype=np.int8))
        f.create_dataset("rewards", data=np.zeros(1, dtype=np.float32))
        f.create_dataset("terminals", data=np.zeros(1, dtype=bool))
        # No attrs at all.
    with pytest.raises(ValueError, match="missing required attribute"):
        load_demo(out)


def test_load_rejects_wrong_format_version(tmp_path):
    obs, actions, rewards, terminals = _make_dummy(1)
    out = tmp_path / "future.h5"
    save_demo(
        out,
        observations=obs, actions=actions, rewards=rewards, terminals=terminals,
        rom_sha256="x", frame_skip=24, start_state="x",
        recorded_at="x", human_player=True,
    )
    # Tamper with the saved file.
    with h5py.File(out, "r+") as f:
        f.attrs["format_version"] = "999"
    with pytest.raises(ValueError, match="unsupported format_version"):
        load_demo(out)


def test_required_attrs_match_save_signature():
    """The set of required attrs documented in the module is the set
    actually written by save_demo. Catches drift if either side moves."""
    written = {
        "format_version", "rom_sha256", "frame_skip",
        "start_state", "recorded_at", "human_player",
    }
    assert set(REQUIRED_ATTRS) == written


def test_sha256_file_is_stable_and_deterministic(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello world")
    a = sha256_file(f)
    b = sha256_file(f)
    assert a == b
    assert len(a) == 64
    # Known SHA-256 of "hello world"
    assert a == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
