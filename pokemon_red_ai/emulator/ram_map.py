"""Pokemon Red WRAM address map and small read helpers.

All addresses are documented in the pret/pokered disassembly
(https://github.com/pret/pokered/blob/master/ram/wram.asm). The names
below match the disassembly's `wFoo` symbols, dropping the `w` prefix.

This module is the single source of truth for "where is X in memory".
Anything that wants raw RAM should go through these constants — never
hardcode an address at the call site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyboy import PyBoy


# --- Party --------------------------------------------------------------

PARTY_COUNT = 0xD163           # number of Pokemon in party (0-6)
PARTY_SPECIES_LIST = 0xD164    # 6 bytes + 0xFF terminator
PARTY_MON_STRUCT = 0xD16B      # start of first party mon (44 bytes each)
PARTY_MON_STRIDE = 44

# Offsets within a single 44-byte party mon struct
MON_SPECIES = 0
MON_CURRENT_HP = 1             # 2 bytes, big-endian
MON_STATUS = 4
MON_TYPE1 = 5
MON_TYPE2 = 6
MON_LEVEL = 33                 # 0x21
MON_MAX_HP = 34                # 2 bytes, big-endian


# --- Player / world -----------------------------------------------------

CURRENT_MAP = 0xD35E
PLAYER_Y = 0xD361
PLAYER_X = 0xD362


# --- Economy / progress -------------------------------------------------

PLAYER_MONEY = 0xD347          # 3 bytes, BCD-encoded, big-endian
OBTAINED_BADGES = 0xD356       # bitfield: bit 0 = Boulder, bit 1 = Cascade, ...
                               #           bit 2 = Thunder, bit 3 = Rainbow,
                               #           bit 4 = Soul,    bit 5 = Marsh,
                               #           bit 6 = Volcano, bit 7 = Earth


# --- Pokedex ------------------------------------------------------------

POKEDEX_OWNED = 0xD2F7         # 19 bytes (152 bits, 151 used)
POKEDEX_SEEN = 0xD30A          # 19 bytes
POKEDEX_BYTES = 19


# --- Battle -------------------------------------------------------------

IS_IN_BATTLE = 0xD057          # 0 = no battle, 1 = wild, 2 = trainer, 0xFF = lost


# --- Event flags --------------------------------------------------------
# 320 bytes covering 2560 individual story flags. Indices for specific
# events (e.g. "got Pokedex", "beat Brock") are pulled from
# pret/pokered/constants/event_constants.asm and added here as we need them.

EVENT_FLAGS = 0xD747
EVENT_FLAGS_BYTES = 320


# --- Helpers ------------------------------------------------------------
# These convert raw bytes into the structured values we'll feed to the
# reward function, metrics, and policy. They take a PyBoy instance so the
# emulator wrapper stays a thin pass-through.


def read_party_levels(pyboy: "PyBoy") -> list[int]:
    """Return the levels of the Pokemon currently in the party.

    Length matches `read_party_count(pyboy)`; empty list if party is empty.
    """
    count = read_party_count(pyboy)
    return [
        pyboy.memory[PARTY_MON_STRUCT + i * PARTY_MON_STRIDE + MON_LEVEL]
        for i in range(count)
    ]


def read_party_count(pyboy: "PyBoy") -> int:
    return min(pyboy.memory[PARTY_COUNT], 6)


def read_party_hp_fractions(pyboy: "PyBoy") -> list[float]:
    """Return current/max HP per party mon, in [0.0, 1.0]. Empty if no party."""
    count = read_party_count(pyboy)
    out: list[float] = []
    for i in range(count):
        base = PARTY_MON_STRUCT + i * PARTY_MON_STRIDE
        cur = _read_u16_be(pyboy, base + MON_CURRENT_HP)
        mx = _read_u16_be(pyboy, base + MON_MAX_HP)
        out.append(cur / mx if mx > 0 else 0.0)
    return out


def read_badges_count(pyboy: "PyBoy") -> int:
    """Number of badges currently earned (popcount of the badges byte)."""
    return bin(pyboy.memory[OBTAINED_BADGES]).count("1")


def has_boulder_badge(pyboy: "PyBoy") -> bool:
    """True once Brock has been beaten — the v1 success signal."""
    return bool(pyboy.memory[OBTAINED_BADGES] & 0x01)


def read_money(pyboy: "PyBoy") -> int:
    """Player money as an integer. Stored on-cart as 3-byte BCD."""
    b0 = pyboy.memory[PLAYER_MONEY]
    b1 = pyboy.memory[PLAYER_MONEY + 1]
    b2 = pyboy.memory[PLAYER_MONEY + 2]
    return _bcd_to_int(b0) * 10000 + _bcd_to_int(b1) * 100 + _bcd_to_int(b2)


def read_pokedex_owned_count(pyboy: "PyBoy") -> int:
    return _popcount_region(pyboy, POKEDEX_OWNED, POKEDEX_BYTES)


def read_pokedex_seen_count(pyboy: "PyBoy") -> int:
    return _popcount_region(pyboy, POKEDEX_SEEN, POKEDEX_BYTES)


def read_event_flags_set_count(pyboy: "PyBoy") -> int:
    """Number of story event flags currently set across the whole flag region.

    This is the fine-grained progress proxy used in the eval rubric.
    """
    return _popcount_region(pyboy, EVENT_FLAGS, EVENT_FLAGS_BYTES)


def read_is_in_battle(pyboy: "PyBoy") -> int:
    return pyboy.memory[IS_IN_BATTLE]


def read_position(pyboy: "PyBoy") -> tuple[int, int, int]:
    """Return (map_id, x, y) for the player."""
    return (
        pyboy.memory[CURRENT_MAP],
        pyboy.memory[PLAYER_X],
        pyboy.memory[PLAYER_Y],
    )


# --- Internal -----------------------------------------------------------


def _read_u16_be(pyboy: "PyBoy", addr: int) -> int:
    return (pyboy.memory[addr] << 8) | pyboy.memory[addr + 1]


def _bcd_to_int(byte: int) -> int:
    return ((byte >> 4) & 0x0F) * 10 + (byte & 0x0F)


def _popcount_region(pyboy: "PyBoy", addr: int, length: int) -> int:
    return sum(bin(pyboy.memory[addr + i]).count("1") for i in range(length))
