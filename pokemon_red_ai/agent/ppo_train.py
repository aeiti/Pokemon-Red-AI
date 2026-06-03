"""PPO training over PokemonRedOverworldEnv with custom v1 eval metrics.

Builds N parallel emulator environments, optionally loads a BC
pretraining checkpoint, runs ``model.learn()``, and periodically runs
a custom eval that computes the rubric scalars from CLAUDE.md:

    eval/brock_success_rate    fraction of K eval episodes that earned
                               the Boulder Badge
    eval/median_steps_to_brock median step count among successful runs
                               (NaN if none succeeded)
    eval/mean_event_flags      mean event-flags-set count at episode
                               end across all eval episodes — the
                               fine-grained progress proxy for runs
                               that didn't beat Brock
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from pokemon_red_ai.env.overworld_env import PokemonRedOverworldEnv


class BrockEvalCallback(BaseCallback):
    """Run K eval episodes every ``eval_freq`` env-steps and log the rubric."""

    def __init__(
        self,
        eval_env: PokemonRedOverworldEnv,
        eval_freq: int,
        n_episodes: int = 20,
        deterministic: bool = True,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self._eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.deterministic = deterministic
        self._next_eval = eval_freq

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps >= self._next_eval:
            self._run_eval()
            self._next_eval = self.num_timesteps + self.eval_freq
        return True

    def _run_eval(self) -> None:
        results: list[dict[str, Any]] = []
        for _ in range(self.n_episodes):
            obs, _info = self._eval_env.reset()
            steps_to_brock: int | None = None
            final_info: dict[str, Any] = {}
            step_count = 0
            while True:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, _reward, terminated, truncated, info = self._eval_env.step(int(action))
                step_count += 1
                final_info = info
                if steps_to_brock is None and info.get("progress/boulder_badge", 0) > 0:
                    steps_to_brock = step_count
                if terminated or truncated:
                    break
            results.append(
                {
                    "boulder_earned": steps_to_brock is not None,
                    "steps_to_brock": steps_to_brock,
                    "event_flags": float(final_info.get("progress/event_flags_set", 0)),
                }
            )

        success_rate = float(np.mean([r["boulder_earned"] for r in results]))
        successful_steps = [r["steps_to_brock"] for r in results if r["boulder_earned"]]
        median_steps = float(np.median(successful_steps)) if successful_steps else float("nan")
        mean_events = float(np.mean([r["event_flags"] for r in results]))

        self.logger.record("eval/brock_success_rate", success_rate)
        self.logger.record("eval/median_steps_to_brock", median_steps)
        self.logger.record("eval/mean_event_flags", mean_events)
        self.logger.dump(self.num_timesteps)

        if self.verbose:
            print(
                f"[eval @ step {self.num_timesteps:>8d}]  "
                f"success={success_rate:.1%}  "
                f"med_steps={median_steps if successful_steps else '-':>6}  "
                f"mean_events={mean_events:.1f}"
            )


def train_ppo(
    output_path: Path | str,
    *,
    bc_checkpoint: Path | str | None = None,
    n_envs: int = 8,
    total_timesteps: int = 1_000_000,
    eval_freq: int = 50_000,
    eval_episodes: int = 20,
    save_freq: int = 100_000,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    learning_rate: float = 3e-4,
    device: str = "auto",
    seed: int = 42,
    tensorboard_dir: Path | str | None = "runs/ppo",
    vec_env_cls: type[VecEnv] = SubprocVecEnv,
    env_kwargs: dict[str, Any] | None = None,
    verbose: int = 1,
) -> dict[str, Any]:
    """Run PPO training and save the resulting model.

    If ``bc_checkpoint`` is given, the policy weights are loaded from
    that SB3 zip (architecture must match). Otherwise a fresh
    ``MlpPolicy`` is initialized.

    Returns metrics dict with the checkpoint path and config summary.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(env_kwargs or {})
    set_random_seed(seed)
    resolved_device = _resolve_device(device)

    vec_env = vec_env_cls([_env_factory(env_kwargs) for _ in range(n_envs)])
    eval_env = PokemonRedOverworldEnv(**env_kwargs)

    tb_log = str(tensorboard_dir) if tensorboard_dir else None

    if bc_checkpoint is not None:
        model = PPO.load(
            str(bc_checkpoint),
            env=vec_env,
            device=resolved_device,
            tensorboard_log=tb_log,
        )
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            device=resolved_device,
            seed=seed,
            tensorboard_log=tb_log,
            verbose=verbose,
        )

    callbacks: list[BaseCallback] = [
        BrockEvalCallback(
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_episodes=eval_episodes,
            verbose=verbose,
        ),
    ]
    if save_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(save_freq // max(n_envs, 1), 1),
                save_path=str(output_path.parent),
                name_prefix=output_path.stem,
            )
        )

    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
        model.save(output_path)
    finally:
        eval_env.close()
        vec_env.close()

    return {
        "checkpoint": str(output_path),
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "device": resolved_device,
        "bc_checkpoint": str(bc_checkpoint) if bc_checkpoint else None,
    }


def _env_factory(env_kwargs: dict[str, Any]) -> Callable[[], PokemonRedOverworldEnv]:
    """Pickle-friendly env constructor for SubprocVecEnv."""

    def _make() -> PokemonRedOverworldEnv:
        return PokemonRedOverworldEnv(**env_kwargs)

    return _make


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
