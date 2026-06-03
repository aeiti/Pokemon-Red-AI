# Pokemon Red AI

Behavioral cloning + reinforcement learning agent that plays Pokemon Red.

See [`CLAUDE.md`](CLAUDE.md) for the full project plan, architecture
decisions, and conventions.

## Status

Early scaffolding. v1 milestone: an agent that beats Brock.

## Setup

Requires [uv](https://github.com/astral-sh/uv) and Python 3.11.

```sh
uv sync
```

Place your legally-obtained Pokemon Red ROM at `roms/pokemon_red.gb`,
or set `POKEMON_RED_ROM` to a path of your choice.

## Layout

- `pokemon_red_ai/emulator/` — PyBoy wrapper and RAM address map
- `pokemon_red_ai/env/` — Gymnasium environment, reward, metrics
- `pokemon_red_ai/agent/` — BC + PPO training
- `scripts/` — CLI entry points (record states, record demos, train, evaluate)
- `saved_states/` — Reproducibility checkpoints (committed)
- `tests/` — Pytest suite

## Running

(Nothing runnable yet — scripts land in subsequent branches.)
