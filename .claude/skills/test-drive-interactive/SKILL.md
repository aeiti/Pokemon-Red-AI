---
name: test-drive-interactive
description: Before merging any code that handles human input — keyboard, mouse, an SDL/GUI window, a CLI launcher the user double-clicks — run a 30-second interactive smoke test with the user. Unit tests miss real-world UX bugs (wrong default key, output going to the wrong stream, paths that crash on absolute inputs, glob ordering that picks the wrong file). Triggers when you've just written or modified an interactive script, a launcher, anything that prompts the user, or anything that opens a window. Do not rely on "the tests pass" as evidence the code works for a human.
---

# Test-drive interactive scripts before merging

Unit tests verify that functions return the right values. They cannot
verify that a human pressing F1 in an SDL window actually causes a
save state to be written. They cannot verify that the SDL window's
title bar reflects what you set, or that the user can read the save
confirmation, or that a path crashes when the user passes `/tmp/`
instead of a repo-relative path.

Every interactive change merged in this session almost shipped a bug
that only an interactive run would catch:

- **PyBoy `tick()` return type.** The 9 unit tests for the env passed.
  The first interactive run failed because we wrote `assert ok is
  True` and `tick()` returns `1`, not `True`.
- **`pyboy.title_status` write attempt.** Pure inspection looked
  settable. The first F1 press in the SDL window crashed AFTER the
  save was written — the file landed, the script died, the user saw
  nothing.
- **`.relative_to(REPO_ROOT)` on a `/tmp/` path.** The CLI parsed fine.
  The smoke test ran fine (output was inside the repo). The user
  passed `/tmp/resume_test.h5` and hit a `ValueError` before the
  emulator even opened.
- **zsh `(NOm)` glob qualifier.** No error. The launcher silently
  loaded the OLDEST save state instead of the newest, corrupting a
  multi-session chain. Took a side-by-side inspector diff hours
  later to notice.

These bugs share two traits:
1. They were invisible to unit tests because the failure mode lived
   in human interaction or in real-world inputs the test fixtures
   didn't model.
2. They were obvious within 30 seconds of an actual interactive run.

## When this skill fires

You wrote, modified, or are about to ship any of:
- A script that opens a window for the user to interact with.
- A keyboard / mouse / gamepad event handler.
- A `.command` / `.desktop` / shortcut launcher.
- A CLI that the user will invoke with paths or values you didn't
  enumerate at test time (e.g., paths outside the project).
- A startup banner / prompt / setup wizard.
- Anything that auto-decides which file/checkpoint/state to load
  ("the latest one", "the most recent run", etc.).

## The procedure

1. **Write your normal unit tests.** Don't skip them. They catch
   different bugs than test-drives do.

2. **Before merging, launch the script in the background.** Use the
   project's actual launch mode — for `.command` files, simulate the
   double-click path (no `~/.zshrc` sourced). For Python scripts,
   `uv run python scripts/<name>.py` from the repo root.

3. **Hand the user a specific 30-second exercise.** Example:
   > Press F1 once on the title screen → expect a `[saved #1]` line.
   > Press F1 again immediately → expect `[saved #2]` with a different timestamp.
   > Hold F1 for one full second → expect exactly one new line, not many.
   > Close the window → expect a clean `Session ended` summary.

   The exercise should:
   - Cover the new code paths
   - Include edge cases (rapid repeats, holds, immediate close)
   - Produce inspectable output (files on disk, stdout lines, window state)

4. **After the user finishes, read every output the script produces.**
   - Read the captured stdout/stderr from the background task.
   - List any files the script was supposed to write.
   - Run any inspector tools (`inspect_state.py`, `inspect_demo.py`,
     etc.) against the outputs.
   - Cross-check what's on disk against what the user said they did.

5. **Compare to expectations specifically.** "It worked" is not
   verification. "The user pressed F1 four times, four `[saved #N]`
   lines appeared, four files exist on disk, each is 167 KB, the
   timestamps are monotonically increasing" — that's verification.

6. **If anything is off, fix on the same branch.** Do not merge
   "and we'll see in production." Fix and re-test-drive.

## Failure-mode lookup

| Symptom in interactive run | Likely cause |
|---|---|
| Files appear but wrong contents | Saved before mutation, or saved a stale snapshot |
| Files do not appear | Path crash, permissions, or wrong directory |
| Files appear in wrong directory | Launcher cwd is not the repo |
| User reports "the same thing as last time" | Auto-resume picked the wrong file (sort order, glob, default fallback) |
| Hotkey does nothing | OS intercepted it (macOS Fn+F1), or vendor consumed the event upstream |
| Hotkey does too much | Vendor has its own binding on that key (see [[audit-vendor-hotkeys]]) |
| Script exits with code 0 but did nothing visible | A `try` swallowed an error, or output went to a closed stream |
| Banner says one thing, behavior is another | Auto-selected input differs from what's printed — make the banner trustworthy |

## When NOT to skip this

- "It's a tiny change" — many of the bugs above were one-line fixes.
- "The tests pass" — none of those tests would have failed; they
  exercised different surfaces.
- "I'll test it after merge" — once it's in `main`, every future
  branch builds on it. Find it now.

The only valid reason to skip is **the user has explicitly said skip
the test-drive for this specific change**. Otherwise, 30 seconds
before merge beats the alternatives.
