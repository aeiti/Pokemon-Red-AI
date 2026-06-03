# Pokemon Red AI — project notes for Claude

This file is the durable plan and conventions for this project. Read it
before making suggestions or changes.

## Goal

Long-term: an agent that beats the Elite Four in Pokemon Red (Game Boy).

This is at the edge of what's been done publicly with RL on this game.
Whidden's well-known agent reached roughly Cerulean (badge 2 area).
We expect the project to take many iterations and ship in milestones.

## v1 milestone — beat Brock

The first complete deliverable is an agent that earns the Boulder Badge
with the full pipeline working end-to-end:

- Save-state checkpoint loaded at episode reset
- RAM-based observation, 9-action discrete space
- Behavioral cloning pretraining on recorded human play
- PPO fine-tuning initialized from BC weights
- TensorBoard scalars + periodic deterministic eval

**v1 success = `eval/brock_success_rate >= 0.8` over a 20-episode eval run.**

Subsequent milestones (v2, v3, …) advance through gyms toward the E4.

## Architecture

- **Two policies, not one.** Overworld and battle are different problems
  (real-time exploration vs turn-based tactics). A small controller
  switches between them based on the game-mode flag in RAM.
- **BC pretraining → PPO fine-tune.** Random exploration cannot discover
  menu navigation or battle UI. Human demonstrations bootstrap a
  competent prior; PPO improves from there.
- **Hierarchical / skill-based for later milestones.** End-to-end PPO
  will plateau. Beating later gyms likely needs scripted or learned
  skills with a high-level objective policy on top. Not in v1.
- **Curriculum via save states.** Train to badge N from a save state
  that has badge N-1 already earned. Avoids re-learning the early game
  every milestone.

## Stack

- **Emulator:** PyBoy 2.x (Python/Cython, scriptable, frame-stepping)
- **RL framework:** Stable-Baselines3 (PPO)
- **Env interface:** Gymnasium
- **Deep learning:** PyTorch (MPS backend on Apple Silicon)
- **Demo storage:** HDF5 (h5py)
- **Logging:** TensorBoard (local; switch to W&B later if desired)
- **Tooling:** uv for env/lock; pytest for tests
- **Python:** 3.11

## Hardware

Apple Silicon M4 Max. Plan for 12–16 parallel PyBoy envs (CPU-bound
emulator) feeding a single PyTorch model on MPS for PPO updates.

## Repo layout

```
pokemon_red_ai/
  emulator/      PyBoy wrapper, RAM address map, state extraction
  env/           Gymnasium env, reward function, metrics
  agent/         BC + PPO training, policy networks
saved_states/    Reproducibility checkpoints (committed; small)
scripts/         CLI entry points: record_state, record_demo, train, evaluate
tests/           Pytest suite — smoke tests for plumbing, not RL training
notebooks/       Exploration, reward tuning, post-hoc analysis
roms/            User-supplied ROM (gitignored)
demonstrations/  Recorded human play traces (gitignored, large)
runs/            TensorBoard logs (gitignored)
checkpoints/     Trained model weights (gitignored)
```

## ROM

User-supplied. Lives at `roms/pokemon_red.gb`. Scripts read from
`POKEMON_RED_ROM` env var with that path as fallback. Never committed.

## Save states

Committed to `saved_states/` because they're small and reproducibility-
critical (every run must start from the same point to compare fairly).
Named by milestone, e.g. `v1_starter_chosen.state`, `v2_post_brock.state`.

## Evaluation rubric

**Per-episode (dense, logged every rollout):**
- `progress/badges` — North Star. 0 or 1 for v1.
- `progress/event_flags_set` — count of story event flags flipped.
  Fine-grained; this is the metric that moves during early training.
- `progress/unique_maps_visited` — exploration breadth
- `progress/max_map_id_reached` — rough depth proxy
- `party/level_sum`, `party/count`, `party/avg_hp_fraction`
- `economy/money` — proxy for trainer wins
- `pokedex/seen`, `pokedex/caught`
- `episode/length_steps`, `episode/total_reward`

**Periodic clean eval (every ~100k PPO steps, K=20 deterministic episodes):**
- `eval/brock_success_rate` — v1 success metric
- `eval/median_steps_to_brock` — efficiency among successful runs
- `eval/mean_event_flags` — progress proxy for failed runs

Implemented in `pokemon_red_ai/env/metrics.py` as a pure function
`metrics(pyboy) -> dict`. Single source of truth for both training
loggers and eval scripts.

## Reward function

Multi-component (Whidden-style). Starting recipe:
- Sum of party levels (training proxy)
- Exploration bonus from screen-hash novelty (visit-new-areas)
- Badge count (sparse, large)
- HP fraction (don't black out)
- Event flags set (story progress)

Expect to iterate. Reward shaping is the actual hard problem in this
project; treat the first version as a starting point, not a target.

## Observation & action space

- **Observation:** RAM-based. Structured features extracted via the RAM
  map (party state, map ID, HP, badges, event flags, mode flag, etc.).
  No raw pixels in v1.
- **Action space:** 9-action discrete — Up, Down, Left, Right, A, B,
  Start, Select, No-op.
- **Frame-skip:** ~24 (decide ~4× per second). Standard for this game;
  agents don't need 60Hz decisions.

## Conventions

### Git

- **Branch per task.** Off `main`, named appropriately
  (`feat/...`, `chore/...`, `fix/...`).
- **One commit per logical change.** Multi-file commits are fine when
  changes are inseparable; split when they're not.
- **No merging to `main` without explicit user OK.**
- All branching/merging local before push to origin.
- Commit messages: imperative mood, short subject, body if non-obvious.

### Code

- Trust internal code; validate only at system boundaries.
- Default to no comments. Comment only when WHY is non-obvious
  (a quirky RAM offset, a workaround for emulator behavior).
- Pin dependency versions for reproducibility.
- Set random seeds at the env boundary; expose them in run configs.

### Tests

- Pytest for plumbing: env conforms to Gymnasium, RAM extraction
  returns sane types/ranges, BC can overfit a tiny synthetic dataset.
- RL training itself isn't unit-testable. Don't try.

## External references

- [PyBoy](https://github.com/Baekalfen/PyBoy) — emulator
- [pret/pokered](https://github.com/pret/pokered) — disassembly,
  source of RAM addresses and event flag IDs
- [PWhiddy/PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments)
  — reference RL implementation, reward shaping ideas
