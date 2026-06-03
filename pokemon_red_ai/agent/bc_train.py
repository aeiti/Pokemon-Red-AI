"""Behavioral cloning over an SB3 PPO policy network.

Builds a PPO model around a shape-only dummy env (so BC training
doesn't require the ROM or a running PyBoy) and trains the policy via
cross-entropy on ``(obs, action)`` pairs from human demonstrations.
The saved checkpoint is an ordinary SB3 model zip — PPO can resume
from it directly in the next branch.

The value head is intentionally left at its random initialization;
it gets trained from rewards once PPO takes over.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter

from pokemon_red_ai.agent.bc_dataset import DemoDataset
from pokemon_red_ai.env.observation import OBSERVATION_DIM
from pokemon_red_ai.env.overworld_env import NUM_ACTIONS


class _ShapeOnlyEnv(gym.Env):
    """Minimal Gym env exposing only the observation/action shapes.

    PPO instantiation reads ``observation_space`` and ``action_space``
    to size its network. We never call reset/step on this env.
    """

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(OBSERVATION_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(OBSERVATION_DIM, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(OBSERVATION_DIM, dtype=np.float32), 0.0, False, False, {}


def train_bc(
    demo_paths: Iterable[Path | str],
    output_path: Path | str,
    *,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.1,
    device: str = "auto",
    seed: int = 42,
    tensorboard_dir: Path | str | None = None,
    verbose: bool = True,
) -> dict:
    """Train a BC policy and save it as an SB3-compatible PPO checkpoint.

    Returns a dict with the final epoch's train/val metrics and the
    checkpoint path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    set_random_seed(seed)
    resolved_device = _resolve_device(device)

    dataset = DemoDataset(demo_paths)
    n_val = max(1, int(round(len(dataset) * val_split)))
    n_train = len(dataset) - n_val
    if n_train < 1:
        raise ValueError(
            f"Dataset has {len(dataset)} samples; val_split={val_split} "
            f"leaves no training data."
        )

    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=gen)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    env = _ShapeOnlyEnv()
    model = PPO("MlpPolicy", env, device=resolved_device, seed=seed, verbose=0)

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)
    writer = SummaryWriter(str(tensorboard_dir)) if tensorboard_dir else None

    train_loss = val_loss = float("nan")
    val_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, resolved_device)
        val_loss, val_acc = _eval_one_epoch(model, val_loader, resolved_device)

        if writer is not None:
            writer.add_scalar("bc/train_loss", train_loss, epoch)
            writer.add_scalar("bc/val_loss", val_loss, epoch)
            writer.add_scalar("bc/val_accuracy", val_acc, epoch)

        if verbose:
            print(
                f"epoch {epoch:>3}/{epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.2%}"
            )

    model.save(output_path)
    if writer is not None:
        writer.close()

    return {
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "checkpoint": str(output_path),
        "device": resolved_device,
        "n_train": n_train,
        "n_val": n_val,
        "demos": [str(p) for p in dataset.paths],
    }


def _train_one_epoch(model, loader: DataLoader, optimizer, device: str) -> float:
    model.policy.train()
    total = 0.0
    n = 0
    for obs, action in loader:
        obs = obs.to(device)
        action = action.to(device)

        _, log_probs, _ = model.policy.evaluate_actions(obs, action)
        loss = -log_probs.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item() * action.shape[0]
        n += action.shape[0]
    return total / max(n, 1)


def _eval_one_epoch(model, loader: DataLoader, device: str) -> tuple[float, float]:
    model.policy.eval()
    total = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for obs, action in loader:
            obs = obs.to(device)
            action = action.to(device)

            _, log_probs, _ = model.policy.evaluate_actions(obs, action)
            loss = -log_probs.mean()

            preds = model.policy.get_distribution(obs).mode()
            correct += int((preds == action).sum().item())

            total += loss.item() * action.shape[0]
            n += action.shape[0]
    return total / max(n, 1), correct / max(n, 1)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
