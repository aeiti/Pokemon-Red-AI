"""Behavioral cloning smoke test.

We can't unit-test RL training in any meaningful sense, but BC is
ordinary supervised learning — if the training loop is sound, it will
overfit a small synthetic dataset where the mapping is deterministic.

We construct observations where the correct action is a clean step
function of a single observation dim, then run the training pipeline
end to end. A correctly wired BC implementation generalizes this
trivially. If this test ever fails, the wiring is broken.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pokemon_red_ai.agent.bc_train import train_bc
from pokemon_red_ai.env.demo_format import save_demo
from pokemon_red_ai.env.observation import OBSERVATION_DIM


def _make_learnable_demo(path: Path, n: int = 500, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    obs = rng.random((n, OBSERVATION_DIM), dtype=np.float32)
    # Deterministic, easy-to-learn mapping: bucket obs[:, 0] into 9 actions.
    # A 2x64 tanh MLP solves this trivially given enough samples.
    actions = np.minimum(np.floor(obs[:, 0] * 9).astype(np.int8), 8)
    save_demo(
        path,
        observations=obs,
        actions=actions,
        rewards=np.zeros(n, dtype=np.float32),
        terminals=np.zeros(n, dtype=bool),
        rom_sha256="x", frame_skip=24, start_state="x",
        recorded_at="x", human_player=True,
    )


def test_bc_overfits_synthetic_dataset(tmp_path):
    demo_path = tmp_path / "synth.h5"
    _make_learnable_demo(demo_path, n=1000, seed=0)

    output = tmp_path / "bc.zip"
    metrics = train_bc(
        [demo_path],
        output,
        epochs=60,
        batch_size=64,
        lr=1e-3,
        val_split=0.2,
        device="cpu",                       # deterministic, no MPS/CUDA quirks
        seed=42,
        tensorboard_dir=None,
        verbose=False,
    )

    assert output.is_file(), "Checkpoint not written"
    # 9-class random baseline is log(9) ≈ 2.20 NLL / 11% accuracy. Bars
    # are set well above both but with generous slack — CPU op variance
    # makes the exact numbers wobble across runs.
    assert metrics["val_loss"] < 1.0, (
        f"val_loss={metrics['val_loss']:.4f} is too close to random (~2.2); "
        f"BC training did not converge."
    )
    assert metrics["val_accuracy"] > 0.6, (
        f"val_accuracy={metrics['val_accuracy']:.2%} is below the bar of 60% "
        f"(random would be ~11%); training may be broken or task too hard."
    )
    assert metrics["n_train"] + metrics["n_val"] == 1000


def test_bc_checkpoint_is_ppo_loadable(tmp_path):
    """The saved file must be loadable as a PPO model — that's the whole point."""
    from stable_baselines3 import PPO

    demo_path = tmp_path / "small.h5"
    _make_learnable_demo(demo_path, n=50, seed=1)
    output = tmp_path / "bc.zip"

    train_bc(
        [demo_path], output,
        epochs=3, batch_size=16, lr=1e-3,
        val_split=0.2, device="cpu", seed=7,
        tensorboard_dir=None, verbose=False,
    )

    model = PPO.load(output, device="cpu")
    # Predict on a fake observation — should return an int in [0, 8].
    obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    assert 0 <= int(action) <= 8


def test_train_bc_rejects_empty_dataset(tmp_path):
    demo_path = tmp_path / "tiny.h5"
    _make_learnable_demo(demo_path, n=2, seed=0)
    # val_split=0.9 leaves <1 training sample
    import pytest
    with pytest.raises(ValueError, match="no training data"):
        train_bc(
            [demo_path], tmp_path / "bc.zip",
            epochs=1, batch_size=1, val_split=0.9,
            device="cpu", seed=0, tensorboard_dir=None, verbose=False,
        )


def test_train_bc_metrics_match_dataset_size(tmp_path):
    demo_path = tmp_path / "fifty.h5"
    _make_learnable_demo(demo_path, n=50, seed=2)
    output = tmp_path / "bc.zip"
    metrics = train_bc(
        [demo_path], output,
        epochs=2, batch_size=16, lr=1e-3, val_split=0.2,
        device="cpu", seed=0, tensorboard_dir=None, verbose=False,
    )
    assert metrics["n_train"] + metrics["n_val"] == 50
    assert metrics["demos"] == [str(demo_path)]
    assert Path(metrics["checkpoint"]) == output
