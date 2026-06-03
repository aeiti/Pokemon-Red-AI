"""Smoke tests for the emulator wrapper + RAM map.

These don't assert game-specific values (we haven't pressed Start yet,
so the title screen is whatever it is) — only that the wiring works
and the RAM reads return values in sane ranges. Any failure here means
something in the address constants or the wrapper is broken.
"""

from __future__ import annotations

from pokemon_red_ai.emulator import ram_map


def test_emulator_ticks_and_memory_is_readable(emulator):
    """The emulator advances and we can read RAM bytes."""
    assert emulator.tick(60)
    # Any RAM byte is in [0, 255]. Pick one we'll use heavily.
    map_id = emulator.memory[ram_map.CURRENT_MAP]
    assert 0 <= map_id <= 255


def test_ram_helpers_return_sane_values(emulator):
    """Run past the title screen, then check helper outputs are coherent."""
    emulator.tick(600)

    party_count = ram_map.read_party_count(emulator.pyboy)
    assert 0 <= party_count <= 6

    levels = ram_map.read_party_levels(emulator.pyboy)
    assert len(levels) == party_count
    assert all(0 <= lvl <= 100 for lvl in levels)

    hp_fractions = ram_map.read_party_hp_fractions(emulator.pyboy)
    assert len(hp_fractions) == party_count
    assert all(0.0 <= f <= 1.0 for f in hp_fractions)

    assert 0 <= ram_map.read_badges_count(emulator.pyboy) <= 8
    assert 0 <= ram_map.read_pokedex_seen_count(emulator.pyboy) <= 151
    assert 0 <= ram_map.read_pokedex_owned_count(emulator.pyboy) <= 151
    assert 0 <= ram_map.read_money(emulator.pyboy) <= 999_999
    assert ram_map.read_is_in_battle(emulator.pyboy) in (0, 1, 2, 0xFF)

    event_count = ram_map.read_event_flags_set_count(emulator.pyboy)
    assert 0 <= event_count <= ram_map.EVENT_FLAGS_BYTES * 8


def test_position_is_a_three_tuple(emulator):
    emulator.tick(60)
    pos = ram_map.read_position(emulator.pyboy)
    assert len(pos) == 3
    assert all(0 <= v <= 255 for v in pos)


def test_button_press_advances_frames(emulator):
    """Pressing a button shouldn't blow up and the emulator keeps running."""
    emulator.tick(120)
    emulator.press("start", hold_frames=4, release_frames=4)
    assert emulator.tick(10)


def test_save_and_load_state_roundtrip(emulator, tmp_path):
    """A loaded state restores the exact memory we saved."""
    emulator.tick(300)
    snapshot_path = tmp_path / "snap.state"
    emulator.save_state(snapshot_path)
    before = bytes(emulator.memory[ram_map.CURRENT_MAP : ram_map.CURRENT_MAP + 16])

    # Mutate state by ticking further, then load and confirm we're back.
    emulator.tick(120)
    emulator.load_state(snapshot_path)
    after = bytes(emulator.memory[ram_map.CURRENT_MAP : ram_map.CURRENT_MAP + 16])
    assert before == after


def test_bcd_helper_pure():
    """BCD decoder is pure — no emulator needed."""
    from pokemon_red_ai.emulator.ram_map import _bcd_to_int

    assert _bcd_to_int(0x00) == 0
    assert _bcd_to_int(0x09) == 9
    assert _bcd_to_int(0x10) == 10
    assert _bcd_to_int(0x42) == 42
    assert _bcd_to_int(0x99) == 99
