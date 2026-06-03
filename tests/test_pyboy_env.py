"""Tests for the emulator wrapper helpers."""

from __future__ import annotations

import sdl2
import pyboy.plugins.window_sdl2 as ws

from pokemon_red_ai.emulator.pyboy_env import disable_state_mutation_keys

STATE_MUTATION_KEYS = (
    sdl2.SDLK_x,
    sdl2.SDLK_z,
    sdl2.SDLK_COMMA,
    sdl2.SDLK_PERIOD,
)

GAMEBOY_KEYS = (
    sdl2.SDLK_UP, sdl2.SDLK_DOWN, sdl2.SDLK_LEFT, sdl2.SDLK_RIGHT,
    sdl2.SDLK_a, sdl2.SDLK_s, sdl2.SDLK_RETURN, sdl2.SDLK_BACKSPACE,
)


def test_disable_removes_state_mutation_keys():
    disable_state_mutation_keys()
    for key in STATE_MUTATION_KEYS:
        assert key not in ws.KEY_UP, f"key 0x{key:x} still bound in KEY_UP"
        assert key not in ws.KEY_DOWN, f"key 0x{key:x} still bound in KEY_DOWN"


def test_disable_preserves_gameboy_button_keys():
    """Arrows, A/B, Start/Select must still work as Game Boy inputs."""
    disable_state_mutation_keys()
    for key in GAMEBOY_KEYS:
        assert key in ws.KEY_DOWN
        assert key in ws.KEY_UP


def test_disable_is_idempotent():
    disable_state_mutation_keys()
    disable_state_mutation_keys()
    disable_state_mutation_keys()           # third call: no errors
    for key in STATE_MUTATION_KEYS:
        assert key not in ws.KEY_UP


def test_escape_still_quits():
    """Esc to close the window stays bound."""
    disable_state_mutation_keys()
    assert sdl2.SDLK_ESCAPE in ws.KEY_UP
