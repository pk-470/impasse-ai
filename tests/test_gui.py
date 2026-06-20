"""Headless GUI-logic tests driving the real GUI methods.

Uses SDL's dummy video/audio drivers (no window) and redirects the recovery
pickle to a temp file, so the select -> complete_move -> update and the
export/undo (load_position) paths are exercised exactly as in a real game.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame as pg
import pytest

import impasse.gui as gui_mod
from impasse.gui import GUI, HEIGHT, INFO_WIDTH, WIDTH


@pytest.fixture
def game(tmp_path, monkeypatch):
    """A headless human-vs-human GUI whose recovery file is redirected to tmp."""
    monkeypatch.setattr(gui_mod, "save_file_path", lambda: tmp_path / "save.p")
    pg.init()
    window = pg.display.set_mode((WIDTH + INFO_WIDTH, HEIGHT))
    yield GUI(window)
    pg.quit()


def _first_slide(game):
    """Return an (origin, target, tag) opening slide as the GUI would select it."""
    origin = next(iter(game.all_legal_moves))
    target, tag = next(iter(game.all_legal_moves[origin].items()))
    return origin, target, tag


def test_complete_move_records_last_move_and_advances_turn(game):
    origin, target, tag = _first_slide(game)
    before_turn = game.turn
    game.select(origin)
    game.select(target)
    assert game.undo_activated
    assert game.last_move_data["tag"] == tag
    assert game.last_move_data["color"] == before_turn
    assert origin in game.last_move_data["cells"]
    assert game.turn != before_turn  # an opening slide passes the turn


def test_undo_restores_position_before_last_move(game):
    origin, target, _ = _first_slide(game)
    before = (game.state.copy(), game.turn, dict(game.checkers_total), game.state_hash)
    game.select(origin)
    game.select(target)
    assert game.state_hash != before[3]
    game.undo_move()
    assert (game.state, game.turn, game.checkers_total, game.state_hash) == before
    assert not game.undo_activated


def test_board_update_renders_without_error(game):
    # Exercises draw_checker (piece_at decode) + info/highlight rendering.
    game.board_update()
