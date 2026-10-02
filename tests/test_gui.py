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
from impasse.play import get_cell_from_mouse
from impasse.position import BLACK, WHITE


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


def test_clock_does_not_overwrite_a_decided_game(tmp_path, monkeypatch):
    """Once someone has won, their remaining time running out must not flip the result."""
    monkeypatch.setattr(gui_mod, "save_file_path", lambda: tmp_path / "save.p")
    pg.init()
    window = pg.display.set_mode((WIDTH + INFO_WIDTH, HEIGHT))
    try:
        game = GUI(window, secs=5)
        # A winning bear off leaves the winner on move.
        game.winner = WHITE
        game.turn = WHITE
        game.times = {WHITE: 2, BLACK: 60}
        for _ in range(4):
            game.update_time()
        assert game.winner == WHITE
    finally:
        pg.quit()


@pytest.mark.parametrize(
    "pos", [(WIDTH + 10, 100), (0, 0), (WIDTH + INFO_WIDTH - 1, HEIGHT - 1)]
)
def test_clicks_outside_the_board_map_to_no_cell(pos):
    assert get_cell_from_mouse(pos) is None


def test_click_on_the_board_maps_to_a_cell():
    assert get_cell_from_mouse((5, HEIGHT - 5)) == (0, 0)
