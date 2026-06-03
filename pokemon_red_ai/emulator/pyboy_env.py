"""Thin wrapper around PyBoy.

Keeps the emulator-facing API in one place so the rest of the project
talks to a stable surface and not the raw PyBoy class.

Headless by default. Pass ``render=True`` only when you want to watch
(interactive recording, debugging). Headless skips the SDL window and
runs the emulator as fast as the host can.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal

from pyboy import PyBoy

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROM_PATH = REPO_ROOT / "roms" / "pokemon_red.gb"

Button = Literal["a", "b", "up", "down", "left", "right", "start", "select"]
ALL_BUTTONS: tuple[Button, ...] = (
    "a", "b", "up", "down", "left", "right", "start", "select",
)


def resolve_rom_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the ROM path: explicit arg > $POKEMON_RED_ROM > repo default.

    Raises FileNotFoundError with a helpful message if nothing resolves.
    """
    if explicit is not None:
        candidate = Path(explicit)
    elif env := os.environ.get("POKEMON_RED_ROM"):
        candidate = Path(env)
    else:
        candidate = DEFAULT_ROM_PATH

    if not candidate.is_file():
        raise FileNotFoundError(
            f"Pokemon Red ROM not found at {candidate}. "
            f"Place it at {DEFAULT_ROM_PATH} or set POKEMON_RED_ROM."
        )
    return candidate


class PyBoyEmulator:
    """A thin, ergonomic wrapper around PyBoy 2.x.

    Use as a context manager to guarantee `stop()` is called:

        with PyBoyEmulator() as emu:
            emu.tick(60)
            print(emu.memory[0xD35E])
    """

    def __init__(
        self,
        rom_path: str | os.PathLike[str] | None = None,
        render: bool = False,
        sound: bool = False,
    ) -> None:
        self.rom_path = resolve_rom_path(rom_path)
        self._pyboy = PyBoy(
            str(self.rom_path),
            window="SDL2" if render else "null",
            sound_emulated=sound,
        )

    # --- lifecycle ------------------------------------------------------

    def stop(self) -> None:
        self._pyboy.stop(save=False)

    def __enter__(self) -> "PyBoyEmulator":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # --- emulator access ------------------------------------------------

    @property
    def pyboy(self) -> PyBoy:
        """Escape hatch for code that needs the raw PyBoy instance."""
        return self._pyboy

    @property
    def memory(self):
        """Direct memory view — `emu.memory[addr]` to read a byte."""
        return self._pyboy.memory

    def tick(self, frames: int = 1, render: bool = False) -> bool:
        """Advance the emulator. Returns False if the user closed the window."""
        return self._pyboy.tick(count=frames, render=render)

    # --- input ----------------------------------------------------------

    def press(self, button: Button, hold_frames: int = 8, release_frames: int = 8) -> None:
        """Press a button for ``hold_frames``, then release for ``release_frames``.

        The release pause matters: Pokemon Red's input handler ignores a
        button that's already down on the previous frame, so back-to-back
        presses of the same button need a gap.
        """
        self._pyboy.button_press(button)
        self.tick(hold_frames)
        self._pyboy.button_release(button)
        self.tick(release_frames)

    # --- save states ----------------------------------------------------

    def save_state(self, path: str | os.PathLike[str]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            self._pyboy.save_state(f)

    def load_state(self, path: str | os.PathLike[str]) -> None:
        with open(path, "rb") as f:
            self._pyboy.load_state(f)


@contextmanager
def open_emulator(**kwargs) -> Iterator[PyBoyEmulator]:
    """Convenience context manager: ``with open_emulator() as emu: ...``"""
    emu = PyBoyEmulator(**kwargs)
    try:
        yield emu
    finally:
        emu.stop()
