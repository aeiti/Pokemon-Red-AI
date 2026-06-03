"""Pure tests for the human-input decoder."""

from __future__ import annotations

import sdl2

from pokemon_red_ai.env.keyboard_action import NOOP_ACTION, decode_action


def test_no_keys_is_noop():
    assert decode_action(set()) == NOOP_ACTION


def test_a_button_priority_over_direction():
    """User priority: A > direction when both held."""
    assert decode_action({sdl2.SDL_SCANCODE_A, sdl2.SDL_SCANCODE_RIGHT}) == 4


def test_buttons_priority_order():
    """A > B > Start > Select."""
    # B beats Start
    assert decode_action({sdl2.SDL_SCANCODE_S, sdl2.SDL_SCANCODE_RETURN}) == 5
    # Start beats Select
    assert decode_action({sdl2.SDL_SCANCODE_RETURN, sdl2.SDL_SCANCODE_BACKSPACE}) == 6
    # A beats all
    assert decode_action({
        sdl2.SDL_SCANCODE_A,
        sdl2.SDL_SCANCODE_S,
        sdl2.SDL_SCANCODE_RETURN,
        sdl2.SDL_SCANCODE_BACKSPACE,
    }) == 4


def test_individual_direction_keys():
    assert decode_action({sdl2.SDL_SCANCODE_UP}) == 0
    assert decode_action({sdl2.SDL_SCANCODE_DOWN}) == 1
    assert decode_action({sdl2.SDL_SCANCODE_LEFT}) == 2
    assert decode_action({sdl2.SDL_SCANCODE_RIGHT}) == 3


def test_individual_button_keys():
    assert decode_action({sdl2.SDL_SCANCODE_A}) == 4
    assert decode_action({sdl2.SDL_SCANCODE_S}) == 5
    assert decode_action({sdl2.SDL_SCANCODE_RETURN}) == 6
    assert decode_action({sdl2.SDL_SCANCODE_BACKSPACE}) == 7


def test_direction_priority_when_multiple_held():
    """Up beats Down (per priority order). Rare in practice but defined."""
    assert decode_action({sdl2.SDL_SCANCODE_UP, sdl2.SDL_SCANCODE_DOWN}) == 0
    assert decode_action({sdl2.SDL_SCANCODE_LEFT, sdl2.SDL_SCANCODE_RIGHT}) == 2


def test_unrelated_keys_are_ignored():
    """A keypress not in the priority list doesn't affect the result."""
    assert decode_action({sdl2.SDL_SCANCODE_F1, sdl2.SDL_SCANCODE_ESCAPE}) == NOOP_ACTION


def test_accepts_any_iterable():
    """decode_action() takes an iterable, not specifically a set."""
    assert decode_action([sdl2.SDL_SCANCODE_A]) == 4
    assert decode_action((sdl2.SDL_SCANCODE_DOWN,)) == 1
