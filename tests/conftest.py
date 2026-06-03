"""Pytest fixtures shared across the suite."""

from __future__ import annotations

import pytest

from pokemon_red_ai.emulator.pyboy_env import PyBoyEmulator, resolve_rom_path


@pytest.fixture(scope="session")
def rom_path():
    """Resolve the Pokemon Red ROM, or skip the whole session if missing.

    Tests that need a real ROM should depend on this fixture; tests that
    are pure logic should not, so the suite stays useful on a fresh clone.
    """
    try:
        return resolve_rom_path()
    except FileNotFoundError as exc:
        pytest.skip(str(exc), allow_module_level=False)


@pytest.fixture
def emulator(rom_path):
    """Fresh headless emulator per test. Auto-stops on teardown."""
    with PyBoyEmulator(rom_path=rom_path) as emu:
        yield emu
