"""PPO training CLI.

Continues training a (BC-pretrained or fresh) policy with PPO over
parallel PokemonRedOverworldEnv instances. Logs the v1 eval rubric to
TensorBoard at ``--eval-freq`` env-steps.

Usage:
    # Fresh PPO, defaults (1M steps, 8 parallel envs)
    uv run python scripts/train_ppo.py --output checkpoints/ppo.zip

    # Initialize from a BC checkpoint, longer run
    uv run python scripts/train_ppo.py \\
        --output checkpoints/ppo.zip \\
        --bc-init checkpoints/bc.zip \\
        --total-timesteps 5_000_000 \\
        --n-envs 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pokemon_red_ai.agent.ppo_train import train_ppo


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/ppo.zip"),
        help="Path for the final SB3 model zip.",
    )
    parser.add_argument(
        "--bc-init", type=Path, default=None,
        help="Optional BC pretraining checkpoint to initialize from.",
    )
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval-freq", type=int, default=50_000,
                        help="Env-steps between custom evals (0 = disable).")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--save-freq", type=int, default=100_000,
                        help="Env-steps between checkpoint saves (0 = disable).")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="PPO rollout buffer size per env.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tensorboard-dir", type=Path, default=Path("runs/ppo"),
        help="TensorBoard log directory. Pass empty string to disable.",
    )
    args = parser.parse_args()

    if args.bc_init is not None and not args.bc_init.is_file():
        raise SystemExit(f"BC checkpoint not found: {args.bc_init}")

    metrics = train_ppo(
        output_path=args.output,
        bc_checkpoint=args.bc_init,
        n_envs=args.n_envs,
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_freq=args.save_freq,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        tensorboard_dir=args.tensorboard_dir if str(args.tensorboard_dir) else None,
    )

    print()
    print(f"Saved final checkpoint: {metrics['checkpoint']}")
    print(f"Trained for {metrics['total_timesteps']:,} env-steps "
          f"({metrics['n_envs']} parallel env(s), device={metrics['device']}).")


if __name__ == "__main__":
    main()
