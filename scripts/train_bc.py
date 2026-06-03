"""Behavioral-cloning training CLI.

Consumes one or more HDF5 demo files and trains an MLP policy
compatible with SB3 PPO initialization.

Usage:
    uv run python scripts/train_bc.py --demos demonstrations/*.h5
    uv run python scripts/train_bc.py --demos demos/run1.h5 demos/run2.h5 \\
        --epochs 30 --batch-size 256 --lr 1e-3 --output checkpoints/bc.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pokemon_red_ai.agent.bc_train import train_bc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--demos", type=Path, nargs="+", required=True,
        help="HDF5 demo files to train on (one or many).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/bc.zip"),
        help="Path for the trained SB3 model zip.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto",
                        help="auto (prefers MPS / CUDA), cpu, mps, cuda.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tensorboard-dir", type=Path, default=Path("runs/bc"),
        help="TensorBoard log directory. Pass empty string to disable.",
    )
    args = parser.parse_args()

    for p in args.demos:
        if not p.is_file():
            raise SystemExit(f"Demo file not found: {p}")

    metrics = train_bc(
        demo_paths=args.demos,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        device=args.device,
        seed=args.seed,
        tensorboard_dir=args.tensorboard_dir if str(args.tensorboard_dir) else None,
    )

    print()
    print(f"Saved checkpoint: {metrics['checkpoint']}")
    print(f"Final val_accuracy: {metrics['val_accuracy']:.2%}")
    print(f"Final val_loss:     {metrics['val_loss']:.4f}")
    print(f"Trained on {metrics['n_train']} samples, validated on {metrics['n_val']}.")


if __name__ == "__main__":
    main()
