"""Decode human keyboard input into a Discrete(9) action.

PyBoy's SDL window handles the actual game input (arrows = D-pad, A
key = A button, etc.) — this module observes the same keys in parallel
to record what the human chose.

Priority order, highest first:
    A button  >  B button  >  Start  >  Select
       >  Up  >  Down  >  Left  >  Right  >  no-op

Buttons outrank directions because button presses are usually
deliberate (open menu, confirm, attack), while a held direction is
often the default state during walking. When the player walks while
mashing A through dialogue, the salient choice each frame is "A".

Pure function — takes a set of pressed SDL scancodes, returns an int.
Tests don't need an emulator or a window.
"""

from __future__ import annotations

from typing import Iterable

import sdl2

# Action index matches PokemonRedOverworldEnv.ACTIONS.
NOOP_ACTION = 8

# Priority-ordered (scancode, action_index) pairs. First match wins.
_PRIORITY: tuple[tuple[int, int], ...] = (
    (sdl2.SDL_SCANCODE_A, 4),           # A button
    (sdl2.SDL_SCANCODE_S, 5),           # B button
    (sdl2.SDL_SCANCODE_RETURN, 6),      # Start
    (sdl2.SDL_SCANCODE_BACKSPACE, 7),   # Select
    (sdl2.SDL_SCANCODE_UP, 0),
    (sdl2.SDL_SCANCODE_DOWN, 1),
    (sdl2.SDL_SCANCODE_LEFT, 2),
    (sdl2.SDL_SCANCODE_RIGHT, 3),
)


def decode_action(pressed: Iterable[int]) -> int:
    """Return the Discrete(9) action for the given set of pressed scancodes."""
    pressed_set = set(pressed)
    for scancode, action in _PRIORITY:
        if scancode in pressed_set:
            return action
    return NOOP_ACTION


def poll_pressed_scancodes() -> set[int]:
    """Snapshot which of our action keys are currently held.

    Uses SDL_GetKeyboardState, which doesn't consume events, so PyBoy's
    own SDL handling for game input is unaffected.
    """
    state = sdl2.SDL_GetKeyboardState(None)
    return {scancode for scancode, _ in _PRIORITY if state[scancode]}
