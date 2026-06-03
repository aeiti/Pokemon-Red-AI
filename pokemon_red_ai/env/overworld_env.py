"""Gymnasium environment for the overworld policy.

Resets from the committed v1 starter save state. Each step advances
the emulator by ``frame_skip`` Game Boy frames; the chosen action is
held for the first third of those frames and released for the rest,
matching the input-handler timing in Pokemon Red.

When the agent walks into a battle, the env auto-mashes A until the
battle ends. This is a placeholder so the overworld policy can train
end-to-end without a battle policy yet — the next branch
(feat/battle-env) will train a real battle policy that replaces this
behavior at deployment time.

Termination:
    terminated  -> Boulder Badge earned (v1 success)
    truncated   -> step counter exceeds ``max_steps``

Both flags also fire if the party is wiped, but we surface that as
truncated since the game auto-respawns at the last Pokémon Center
rather than ending an episode in the Gym sense. A clean way to detect
"wiped" deterministically is left for a follow-up reward iteration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from pokemon_red_ai.emulator import ram_map
from pokemon_red_ai.emulator.pyboy_env import PyBoyEmulator
from pokemon_red_ai.env.accumulator import EpisodeAccumulator
from pokemon_red_ai.env.metrics import metrics
from pokemon_red_ai.env.observation import OBSERVATION_DIM, build_observation
from pokemon_red_ai.env.reward import compute_overworld_reward

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = REPO_ROOT / "saved_states" / "v1_starter_chosen.state"

# Action index -> PyBoy button name, or None for no-op.
ACTIONS: tuple[str | None, ...] = (
    "up", "down", "left", "right", "a", "b", "start", "select", None,
)
NUM_ACTIONS = len(ACTIONS)

DEFAULT_FRAME_SKIP = 24
DEFAULT_MAX_STEPS = 2048
BATTLE_RESOLUTION_CAP_FRAMES = 60 * 60   # 60s of simulated time as a safety cap


class PokemonRedOverworldEnv(gym.Env):
    """Single-policy overworld env. Auto-resolves battles with A-mash."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        rom_path: str | os.PathLike[str] | None = None,
        save_state_path: str | os.PathLike[str] | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.save_state_path = Path(save_state_path) if save_state_path else DEFAULT_STATE
        if not self.save_state_path.is_file():
            raise FileNotFoundError(
                f"Save state required at {self.save_state_path}. "
                f"Record one with scripts/record_state.py."
            )

        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(OBSERVATION_DIM,), dtype=np.float32,
        )

        self._emu = PyBoyEmulator(rom_path=rom_path, render=(render_mode == "human"))
        self._accumulator = EpisodeAccumulator()
        self._last_metrics: dict[str, float] | None = None

    # --- Gymnasium API ---------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._emu.load_state(self.save_state_path)
        self._emu.tick(1)
        self._accumulator.reset()
        self._last_metrics = metrics(self._emu.pyboy)
        return build_observation(self._emu.pyboy), self._build_info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._last_metrics is None:
            raise RuntimeError("step() called before reset().")

        self._execute_action(int(action))
        self._auto_resolve_battle()
        self._accumulator.update(self._emu.pyboy)

        current = metrics(self._emu.pyboy)
        reward = compute_overworld_reward(self._last_metrics, current)
        self._last_metrics = current

        terminated = bool(ram_map.has_boulder_badge(self._emu.pyboy))
        truncated = self._accumulator.steps >= self.max_steps

        return (
            build_observation(self._emu.pyboy),
            float(reward),
            terminated,
            truncated,
            self._build_info(),
        )

    def close(self) -> None:
        self._emu.stop()

    # --- Internals -------------------------------------------------------

    def _execute_action(self, action: int) -> None:
        button = ACTIONS[action]
        if button is None:
            self._emu.tick(self.frame_skip)
            return
        hold = max(1, self.frame_skip // 3)
        rest = self.frame_skip - hold
        self._emu.pyboy.button_press(button)
        self._emu.tick(hold)
        self._emu.pyboy.button_release(button)
        self._emu.tick(rest)

    def _auto_resolve_battle(self) -> None:
        """Mash A until the battle flag clears. Bounded to prevent loops."""
        frames = 0
        pyboy = self._emu.pyboy
        while ram_map.read_is_in_battle(pyboy) != 0 and frames < BATTLE_RESOLUTION_CAP_FRAMES:
            pyboy.button_press("a")
            self._emu.tick(8)
            pyboy.button_release("a")
            self._emu.tick(16)
            frames += 24

    def _build_info(self) -> dict[str, Any]:
        return {**metrics(self._emu.pyboy), **self._accumulator.snapshot()}
