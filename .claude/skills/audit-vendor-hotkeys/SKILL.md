---
name: audit-vendor-hotkeys
description: Before recording human input or other state from a third-party interactive tool (emulator, simulator, IDE, embedded app), audit the tool's default keymap for "quality-of-life" bindings that mutate state — save/load slots, rewind, reset, undo, snapshot. Neutralize those bindings for the recording session so a stray press can't silently corrupt the captured trajectory. Triggers when wrapping a vendored interactive tool for data collection, when a recorded trajectory has unexpected state changes the user didn't intend, or when integrating any third-party SDL/Qt/native-window tool that the human will interact with.
---

# Audit vendor hotkeys before recording human input

Third-party interactive tools (emulators, simulators, debuggers,
embedded apps) usually ship with default keybindings the wrapper
inherits silently. Many of these are state-mutating QoL bindings —
save/load slots, rewind, undo, reset, screenshot — that the vendor
finds useful but that **silently corrupt a captured trajectory**
when a human accidentally hits them mid-recording.

The failure mode is invisible: the recorder logs everything correctly,
but the in-game state quietly jumps. You only notice when something
downstream (BC training, replay, statistics) looks weird and you
trace it back. By then the recording is hours old.

## The pattern that catches it

When you wrap any vendored interactive tool for human recording:

1. **Find the vendor's default keymap.** It's almost always defined
   as a module-level or class-level dict. `grep -i 'sdlk_\|key_\|keymap'`
   inside the installed package usually finds it in one shot. For
   Cython-compiled modules where `inspect.getsource` fails, read the
   `.py`/`.pyx` source directly from the venv's site-packages.

2. **Categorize every binding by failure mode:**
   - **State-mutating** (save, load, rewind, undo, reset, screenshot
     overwrite): MUST be disabled for recording. Silent corruption.
   - **Performance/cosmetic** (turbo, pause, fullscreen, screenshot
     to new file, audio toggle): usually safe to leave.
   - **The game/tool's own input** (D-pad, buttons, menus): MUST be
     preserved — it's what the human is using to play.
   - **Quit / window-close** (Esc, ⌘Q): leave alone; the user needs
     a way out.

3. **Patch them out before opening the window.** If keymap is a dict,
   `.pop(key, None)` is safe and idempotent. If it's class-level, the
   patch is process-global — fine for a recorder script.

4. **Mention the patch in the recorder's startup banner.** Future
   users (including future you) will see exactly which keys are no-ops
   and won't waste time pressing them.

5. **Write a test** that asserts the dangerous keys are gone AND the
   tool's normal-input keys are preserved. Catches drift if the vendor
   adds a new binding in an upgrade.

## Concrete examples by tool family

- **PyBoy**: module-level `KEY_UP` / `KEY_DOWN` dicts in
  `pyboy.plugins.window_sdl2`. Risky on KEY_UP: SDLK_x (STATE_LOAD),
  SDLK_z (STATE_SAVE), SDLK_COMMA / SDLK_PERIOD (REWIND).
- **mGBA / Bizhawk via scripting**: check the input config XML/JSON
  for `Reset`, `Save State 1..10`, `Load State 1..10`, `Rewind`.
- **PyGame-based simulators**: usually no built-in state-mutation
  hotkeys — but check anyway.
- **VS Code / JetBrains as an embedded surface**: command palette
  shortcuts (⌘P, ⌘⇧P) can route to anything. Document or shadow.

## When this skill should fire

- About to write a script that opens a third-party interactive window
  for human use.
- A recorded trajectory shows unexplained state jumps (party loss,
  level reset, position teleport, undo of recent edits).
- Wrapping any new vendored tool for the first time — audit the
  default bindings before any user-facing scripts ship.

## How NOT to fix it

- **Don't filter SDL events globally.** PyBoy (or whatever tool) is
  also consuming events from the same queue; intercepting upstream
  steals the inputs you actually want.
- **Don't rely on user discipline.** "Just don't press X" is never a
  fix; the keys are millimeters from keys the user IS supposed to
  press.
- **Don't fix it only in the recorder script.** If the helper is
  reusable, put it in the emulator/tool wrapper module so future
  scripts get it for free.
