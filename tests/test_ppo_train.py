"""PPO training smoke test.

Real PPO learning is not unit-testable — we don't assert anything
about reward going up. The smoke test verifies the train_ppo() wiring:
env factory, model construction, rollout collection, gradient update,
checkpoint save, and PPO.load round-trip.

Uses DummyVecEnv (single-process) so subprocess teardown can't hang
the suite, n_envs=1, and a tiny total_timesteps with matching n_steps
so PPO completes at least one rollout + update.

Skips cleanly if the ROM or v1 starter state is missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from pokemon_red_ai.agent.ppo_train import train_ppo
from pokemon_red_ai.env.overworld_env import DEFAULT_STATE

if not DEFAULT_STATE.is_file():
    pytest.skip(
        f"v1 starter state missing at {DEFAULT_STATE}",
        allow_module_level=True,
    )


def test_train_ppo_smoke_runs_and_saves(rom_path, tmp_path):
    output = tmp_path / "ppo.zip"

    metrics = train_ppo(
        output_path=output,
        bc_checkpoint=None,
        n_envs=1,
        total_timesteps=32,            # tiny: one rollout + one update
        eval_freq=10**9,               # skip eval inside smoke test
        eval_episodes=1,
        save_freq=0,                   # no periodic checkpoint callback
        n_steps=16,
        batch_size=8,
        n_epochs=1,
        device="cpu",
        seed=0,
        tensorboard_dir=None,
        vec_env_cls=DummyVecEnv,
        env_kwargs={
            "rom_path": str(rom_path),
            "max_steps": 16,
            "frame_skip": 4,
        },
        verbose=0,
    )

    assert output.is_file(), "Final checkpoint not written"
    assert metrics["checkpoint"] == str(output)
    assert metrics["total_timesteps"] == 32
    assert metrics["n_envs"] == 1
    assert metrics["bc_checkpoint"] is None


def test_train_ppo_checkpoint_is_loadable(rom_path, tmp_path):
    output = tmp_path / "ppo.zip"
    train_ppo(
        output_path=output,
        n_envs=1, total_timesteps=16,
        eval_freq=10**9, save_freq=0,
        n_steps=8, batch_size=4, n_epochs=1,
        device="cpu", seed=1, tensorboard_dir=None,
        vec_env_cls=DummyVecEnv,
        env_kwargs={"rom_path": str(rom_path), "max_steps": 8, "frame_skip": 4},
        verbose=0,
    )

    model = PPO.load(output, device="cpu")
    assert model is not None
    assert model.policy is not None


def test_train_ppo_resumes_from_bc_checkpoint(rom_path, tmp_path):
    """A BC checkpoint feeds directly into train_ppo() without re-architecting."""
    # Build a BC checkpoint inline via the BC pipeline so this test is
    # independent of any pre-existing files.
    import numpy as np
    from pokemon_red_ai.agent.bc_train import train_bc
    from pokemon_red_ai.env.demo_format import save_demo
    from pokemon_red_ai.env.observation import OBSERVATION_DIM

    rng = np.random.default_rng(0)
    n = 60
    save_demo(
        tmp_path / "demo.h5",
        observations=rng.random((n, OBSERVATION_DIM), dtype=np.float32),
        actions=np.minimum(np.floor(rng.random(n) * 9).astype(np.int8), 8),
        rewards=np.zeros(n, dtype=np.float32),
        terminals=np.zeros(n, dtype=bool),
        rom_sha256="x", frame_skip=24, start_state="x",
        recorded_at="x", human_player=True,
    )

    bc_ckpt = tmp_path / "bc.zip"
    train_bc(
        [tmp_path / "demo.h5"], bc_ckpt,
        epochs=2, batch_size=16, lr=1e-3, val_split=0.2,
        device="cpu", seed=0, tensorboard_dir=None, verbose=False,
    )

    ppo_ckpt = tmp_path / "ppo.zip"
    metrics = train_ppo(
        output_path=ppo_ckpt,
        bc_checkpoint=bc_ckpt,
        n_envs=1, total_timesteps=16,
        eval_freq=10**9, save_freq=0,
        n_steps=8, batch_size=4, n_epochs=1,
        device="cpu", seed=0, tensorboard_dir=None,
        vec_env_cls=DummyVecEnv,
        env_kwargs={"rom_path": str(rom_path), "max_steps": 8, "frame_skip": 4},
        verbose=0,
    )

    assert metrics["bc_checkpoint"] == str(bc_ckpt)
    assert ppo_ckpt.is_file()
    PPO.load(ppo_ckpt, device="cpu")            # loads cleanly
