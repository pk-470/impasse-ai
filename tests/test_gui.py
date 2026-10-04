"""
Headless GUI-logic tests driving the real GUI methods.

Uses SDL's dummy video/audio drivers (no window) and redirects the recovery
pickle to a temp file, so the select -> complete_move -> update and the
export/undo (load_position) paths are exercised exactly as in a real game.
"""

import os
from collections.abc import Iterator
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame as pg
import pytest

import impasse.gui as gui_mod
from impasse.gui import GUI, HEIGHT, INFO_WIDTH, WIDTH
from impasse.play import get_cell_from_mouse
from impasse.position import BLACK, WHITE, Cell


@pytest.fixture
def game(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[GUI]:
    """A headless human-vs-human GUI whose recovery file is redirected to tmp."""
    monkeypatch.setattr(gui_mod, "save_file_path", lambda: tmp_path / "save.p")
    pg.init()
    window = pg.display.set_mode((WIDTH + INFO_WIDTH, HEIGHT))
    yield GUI(window)
    pg.quit()


def _first_slide(game: GUI) -> tuple:
    """An (origin, target, tag) opening slide, as the GUI would select it."""
    origin = next(iter(game.all_legal_moves))
    target, tag = next(iter(game.all_legal_moves[origin].items()))
    return origin, target, tag


class TestCompleteMove:
    def test_records_last_move_and_advances_turn(self, game: GUI) -> None:
        origin, target, tag = _first_slide(game)
        before_turn = game.turn
        game.select(origin)
        game.select(target)
        assert game.undo_activated
        assert game.last_move_data["tag"] == tag
        assert game.last_move_data["color"] == before_turn
        assert origin in game.last_move_data["cells"]
        assert game.turn != before_turn  # an opening slide passes the turn


class TestUndoMove:
    def test_restores_the_position_before_the_last_move(self, game: GUI) -> None:
        origin, target, _ = _first_slide(game)
        before = (
            game.state.copy(),
            game.turn,
            dict(game.checkers_total),
            game.state_hash,
        )
        game.select(origin)
        game.select(target)
        assert game.state_hash != before[3]
        game.undo_move()
        assert (game.state, game.turn, game.checkers_total, game.state_hash) == before
        assert not game.undo_activated


class TestBoardUpdate:
    def test_renders_without_error(self, game: GUI) -> None:
        game.board_update()


class TestUpdateTime:
    def test_does_not_overwrite_a_decided_game(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A winning bear off leaves the winner on move, so their clock keeps running."""
        monkeypatch.setattr(gui_mod, "save_file_path", lambda: tmp_path / "save.p")
        pg.init()
        window = pg.display.set_mode((WIDTH + INFO_WIDTH, HEIGHT))
        try:
            game = GUI(window, secs=5)
            game.winner = WHITE
            game.turn = WHITE
            game.times = {WHITE: 2, BLACK: 60}
            for _ in range(4):
                game.update_time()
            assert game.winner == WHITE
        finally:
            pg.quit()


class TestGetCellFromMouse:
    @pytest.mark.parametrize(
        "pos,cell",
        [
            ((5, HEIGHT - 5), (0, 0)),
            ((WIDTH + 10, 100), None),
            ((0, 0), None),
            ((WIDTH + INFO_WIDTH - 1, HEIGHT - 1), None),
        ],
    )
    def test_maps_a_pixel_to_a_cell_or_none(
        self, pos: tuple[int, int], cell: Cell | None
    ) -> None:
        assert get_cell_from_mouse(pos) == cell
