import pickle
from functools import cache
from pathlib import Path
from typing import Optional, TypedDict

import pygame as pg
from platformdirs import user_data_path

from impasse.ai import AI
from impasse.position import (
    BLACK,
    OPPOSITE_COLOR,
    WHITE,
    Cell,
    Color,
    MoveTag,
    Position,
    _make_state_hash,
    _to_flat,
)


class LastMoveData(TypedDict):
    """The cells touched by the last move, the side that made it, and its tag."""

    cells: list[Optional[Cell]]
    color: Optional[Color]
    tag: Optional[MoveTag]


WIDTH, HEIGHT = 640, 640
SQUARE_SIZE = WIDTH // 8
RADIUS = 2 * SQUARE_SIZE // 5

INFO_WIDTH = 2 * WIDTH // 5
INFO_HEIGHT_PLACEMENT = {WHITE: HEIGHT // 2, BLACK: 0}
COLOR_NAME = {WHITE: "WHITE", BLACK: "BLACK"}

LIGHT = (240, 240, 210)
DARK = (120, 160, 80)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 234, 0)
ORANGE = (255, 140, 0)

COLUMN_COORDS_NUMBERS = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}


def square_draw_tuple(cell: Cell) -> tuple[int, int, int, int]:
    """Return the (x, y, width, height) pixel rectangle of a board cell."""
    return (
        cell[0] * SQUARE_SIZE,
        HEIGHT - (cell[1] + 1) * SQUARE_SIZE,
        SQUARE_SIZE,
        SQUARE_SIZE,
    )


def info_box_draw_tuple(color: Color) -> tuple[int, int, int, int]:
    """Return the (x, y, width, height) pixel rectangle of a player's info box."""
    return (
        WIDTH,
        INFO_HEIGHT_PLACEMENT[color],
        INFO_WIDTH,
        HEIGHT // 2,
    )


def calculate_coords(cell: Cell) -> tuple[float, float]:
    """Return the pixel coordinates of a board cell's centre."""
    return (
        SQUARE_SIZE * (cell[0] + 0.5),
        HEIGHT - (SQUARE_SIZE * (int(cell[1]) + 0.5)),
    )


def cell_to_string(cell: Optional[Cell]) -> Optional[str]:
    """Return a cell's board coordinate (e.g. "C4"), or None if cell is None."""
    if cell:
        return f"{COLUMN_COORDS_NUMBERS[cell[0]]}{cell[1] + 1}"
    return None


@cache
def save_file_path() -> Path:
    """Return the recovery-save file path, creating its directory if needed."""
    return user_data_path("impasse", ensure_exists=True) / "last_position_data.p"


class GUI(Position):
    """A pygame wrapper over Position that plays the game graphically."""

    def __init__(
        self,
        window: pg.Surface,
        secs: Optional[int] = None,
        ai_player: Optional[Color] = None,
        dev: bool = False,
    ) -> None:
        """
        Set up the window, fonts, and a fresh game.

        Args:
            window: The pygame surface to draw on.
            secs: Per-player time budget in seconds for a timed game, or None.
            ai_player: The colour the AI plays, or None for human-vs-human.
            dev: When True, the AI prints per-move search diagnostics.
        """
        pg.init()
        self.window = window
        self.dev = dev
        self.timed = True if secs and not ai_player else False
        self.fonts = {
            "info": pg.font.SysFont("georgia", 24),
            "cell": pg.font.SysFont("georgia", 14),
        }
        self.new_game(secs, ai_player)

    def new_game(self, secs: Optional[int], ai_player: Optional[Color]) -> None:
        """Start a new game from the opening position with the given clock and AI side."""
        self.make_position()
        self.last_move_data: LastMoveData = {"cells": [], "color": None, "tag": None}
        self.undo_activated = False
        self.times: dict[Color, Optional[int]] = {WHITE: secs, BLACK: secs}
        self.export_position_data()
        self.selection_activated = True
        self.selected = None
        self.show_cells = True
        self.ai_player: Optional[AI] = (
            AI(ai_player, dev=self.dev) if ai_player else None
        )
        self._last_tick_draw = 0
        if self.ai_player is not None:
            self.ai_player.on_tick = self.search_tick
        self.print_intro_message()
        if ai_player == WHITE:
            self.ai_play_turn_full()

    def print_intro_message(self):
        """Print the welcome banner to the console."""
        print("--------------------------------------------------")
        print()
        print("Welcome to Impasse!")
        print()
        print("--------------------------------------------------")
        print()

    def export_position_data(self):
        """Save the current game state to the recovery file."""
        position_data = {
            "state": self.state,
            "turn": self.turn,
            "all_legal_moves": self.all_legal_moves,
            "checkers_total": self.checkers_total,
            "winner": self.winner,
            "last_move_data": self.last_move_data,
            "undo_activated": self.undo_activated,
            "times": self.times,
        }
        with open(save_file_path(), "wb") as file:
            pickle.dump(position_data, file)

    def load_position(self):
        """Restore the game state from the recovery file."""
        with open(save_file_path(), "rb") as file:
            position_data = pickle.load(file)
        # _to_flat accepts both the current flat board and a legacy dict board from
        # an older recovery file; the hash is recomputed to stay consistent with it.
        self.state = _to_flat(position_data["state"])
        self.state_hash = _make_state_hash(self.state)
        self.turn = position_data["turn"]
        self.all_legal_moves = position_data["all_legal_moves"]
        self.checkers_total = position_data["checkers_total"]
        self.winner = position_data["winner"]
        self.last_move_data = position_data["last_move_data"]
        self.undo_activated = position_data["undo_activated"]
        self.times = position_data["times"]
        self.selection_activated = not self.winner
        self.selected = None

    def update_time(self) -> None:
        """
        Count down the mover's clock by one second, ending the game in the opponent's favour if it reaches zero
        (called once a second). A decided game's clock is left alone.
        """
        if self.timed and not self.winner:
            time_left = self.times[self.turn]
            if time_left is not None and time_left > 0:
                self.times[self.turn] = time_left - 1
                if time_left == 1:
                    self.winner = OPPOSITE_COLOR[self.turn]

    # Drawing functions

    def board_update(self):
        """Redraw the board, the info boxes, and the move/selection highlights."""
        self.draw_board()
        self.show_info()
        self.show_cell_names()
        if self.winner:
            self.show_winner()
            self.selection_activated = False
        else:
            self.show_last_move()
            self.show_checkers_that_can_move()
            self.show_selected()
            self.show_legal_moves_for_selected()
        pg.display.update()

    def change_show_cells(self) -> None:
        """Toggle whether cell coordinate labels are drawn."""
        self.show_cells = not self.show_cells

    def draw_checker(self, cell: Cell) -> None:
        """Draw the checker occupying a cell, with an inner ring for a double."""
        x, y = calculate_coords(cell)
        piece = self.piece_at(cell)
        assert piece is not None
        color, piece_type = piece
        pg.draw.circle(self.window, color, (x, y), RADIUS)
        if piece_type == 2:
            pg.draw.circle(self.window, OPPOSITE_COLOR[color], (x, y), 2 * RADIUS / 3)
            pg.draw.circle(self.window, color, (x, y), RADIUS / 2)

    def draw_board(self):
        """Draw the 8x8 board squares and the checkers standing on them."""
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    pg.draw.rect(self.window, DARK, square_draw_tuple((i, j)))
                    if self.is_occupied((i, j)):
                        self.draw_checker((i, j))
                else:
                    pg.draw.rect(self.window, LIGHT, square_draw_tuple((i, j)))

    def make_time_string(self, time: int) -> str:
        """Return a whole number of seconds formatted as "mm:ss"."""
        mins = time // 60
        mins = "0" + str(mins) if mins < 10 else str(mins)
        secs = time % 60
        secs = "0" + str(secs) if secs < 10 else str(secs)
        return mins + ":" + secs

    def make_last_move_string(self) -> str:
        """Return the last move as "<from> to <to> (<tag>)"."""
        cells = [cell_to_string(cell) for cell in self.last_move_data["cells"]]
        cell_strings = [cell for cell in cells if cell is not None]
        return " to ".join(cell_strings) + f" ({self.last_move_data['tag']})"

    def make_info_box(self, color: Color) -> None:
        """Draw a player's info box: checker count, clock (if timed), and last move."""
        pg.draw.rect(self.window, color, info_box_draw_tuple(color))
        checkers = self.checkers_total[color]
        # Checkers count
        if checkers and (
            not self.timed or self.timed and self.times[OPPOSITE_COLOR[color]]
        ):
            checkers_num = self.fonts["info"].render(
                f"{COLOR_NAME[color]}: {checkers}",
                True,
                OPPOSITE_COLOR[color],
            )
            self.window.blit(checkers_num, (WIDTH, INFO_HEIGHT_PLACEMENT[color]))
        # Time
        if self.timed:
            time_left = self.times[color]
            assert time_left is not None
            time = self.fonts["info"].render(
                self.make_time_string(time_left),
                True,
                OPPOSITE_COLOR[color],
            )
            self.window.blit(
                time,
                (
                    WIDTH + INFO_WIDTH - time.get_width(),
                    INFO_HEIGHT_PLACEMENT[color] + HEIGHT // 2 - time.get_height(),
                ),
            )
        # Last move
        if self.last_move_data["color"] == color:
            last_move = self.fonts["info"].render(
                self.make_last_move_string(), True, OPPOSITE_COLOR[color]
            )
            self.window.blit(
                last_move,
                (
                    WIDTH,
                    INFO_HEIGHT_PLACEMENT[color] + last_move.get_height(),
                ),
            )

    def show_cell_names(self):
        """Draw each cell's coordinate label when labels are enabled."""
        if self.show_cells:
            for i in range(8):
                for j in range(8):
                    color = LIGHT if (i + j) % 2 == 0 else DARK
                    img = self.fonts["cell"].render(cell_to_string((i, j)), True, color)
                    self.window.blit(img, square_draw_tuple((i, j)))

    def show_winner(self) -> None:
        """Draw the "<colour> WINS!!!" banner for the winning side."""
        assert self.winner is not None
        img = self.fonts["info"].render(
            f"{COLOR_NAME[self.winner]} WINS!!!",
            True,
            OPPOSITE_COLOR[self.winner],
        )
        self.window.blit(img, (WIDTH, INFO_HEIGHT_PLACEMENT[self.winner]))

    def show_info(self):
        """Draw both players' info boxes."""
        self.make_info_box(WHITE)
        self.make_info_box(BLACK)

    def highlight(self, cell: Cell, color: Color) -> None:
        """Draw a small dot of the given colour at a cell's centre."""
        x, y = calculate_coords(cell)
        pg.draw.circle(self.window, color, (x, y), SQUARE_SIZE // 8)

    def show_checkers_that_can_move(self):
        """Highlight the movable checkers, except during the AI's turn."""
        if not (self.ai_player and self.turn == self.ai_player.color):
            for cell in self.all_legal_moves:
                self.highlight(cell, ORANGE)

    def show_legal_moves_for_selected(self):
        """Highlight the destinations of the currently selected checker."""
        if self.selected:
            for move in self.all_legal_moves[self.selected]:
                if move:
                    self.highlight(move, BLUE)

    def show_selected(self):
        """Highlight the currently selected checker."""
        if self.selected:
            self.highlight(self.selected, RED)

    def show_last_move(self):
        """Highlight the cells touched by the last move."""
        for cell in self.last_move_data["cells"]:
            self.highlight(cell, YELLOW)

    # Gameplay functions

    def select(self, cell: Cell) -> None:
        """
        A function that determines what happens when you select a cell (selects checker
        or moves previously selected checker to newly selected cell).
        """
        if not self.selection_activated or self.winner:
            return
        if last_selected := self.selected:
            # If there is a selected cell already, then that cell contains a checker
            # that can be moved on the board. If moving to the newly selected cell is
            # legal, then move. Otherwise, unselect the previous cell and try to select
            # the new one.
            self.selected = None
            if cell in self.all_legal_moves[last_selected]:
                tag = self.all_legal_moves[last_selected][cell]
                self.complete_move(last_selected, cell, tag)
            else:
                self.select(cell)
        # Check if the cell contains a checker that can be moved
        elif cell in self.all_legal_moves:
            # Bear off
            if self.all_legal_moves[cell].get(None) == "B":
                self.complete_move(cell, None, "B")
            # Select cell
            else:
                self.selected = cell

    def complete_move(self, origin: Cell, target: Optional[Cell], tag: MoveTag) -> None:
        """
        Apply a move made on the board, recording it for display and recovery.

        Args:
            origin: The moving checker's cell.
            target: The destination cell, or None for a bear off.
            tag: The move tag.
        """
        self.undo_activated = True
        if not (self.ai_player and self.ai_player.color == self.turn):
            self.export_position_data()
        state_update = self.apply_move(origin, target, tag)
        self.last_move_data = {
            "cells": list(state_update),
            "color": self.turn,
            "tag": tag,
        }
        self.update(state_update, tag)

    def undo_move(self) -> None:
        """
        Restore the board to the state before the last human move (one move only),
        resuming the AI if the restored position is on the AI's turn.
        """
        if self.undo_activated:
            self.load_position()
            self.undo_activated = False
            if (
                self.ai_player is not None
                and not self.winner
                and self.turn == self.ai_player.color
            ):
                self.ai_play_turn_full()

    def change_turn(self) -> None:
        """Pass the turn, printing the post-move evaluation and letting the AI move when it is its turn."""
        Position.change_turn(self)
        if not self.ai_player:
            print("Last move:", self.make_last_move_string())
            print(f"Evaluation after move: {self.evaluate()}")
            print()
            print("--------------------------------------------------")
            print()
        elif (
            self.ai_player is not None
            and not self.winner
            and self.turn == self.ai_player.color
        ):
            self.ai_play_turn_full()

    def search_tick(self) -> None:
        """
        Keep the window responsive while the AI searches.

        Called by the AI every few thousand nodes: pumps the event queue so the OS
        does not mark the window unresponsive, and refreshes the AI's info box with
        the depth in progress (at most five times a second).
        """
        pg.event.pump()
        assert self.ai_player is not None
        now = pg.time.get_ticks()
        if now - self._last_tick_draw < 200:
            return
        self._last_tick_draw = now
        color = self.ai_player.color
        self.make_info_box(color)
        img = self.fonts["info"].render(
            f"thinking... depth {self.ai_player.current_depth}",
            True,
            OPPOSITE_COLOR[color],
        )
        self.window.blit(img, (WIDTH, INFO_HEIGHT_PLACEMENT[color] + HEIGHT // 4))
        pg.display.update(info_box_draw_tuple(color))

    def ai_play_turn(self) -> None:
        """
        Plays a full turn (which may consist of multiple moves) for the AI.
        Also prints the best move suggested by the AI and the evaluation after
        the move (along with the evaluations printed by the suggested_move
        function).
        """
        self.selection_activated = False
        self.board_update()
        print("AI calculating...")
        print()
        assert self.ai_player is not None
        origin, target, tag, unique_move = self.ai_player.suggested_move(self)
        assert origin is not None and tag is not None
        self.complete_move(origin, target, tag)
        print("Best move:", self.make_last_move_string())
        print(f"Evaluation after move: {self.evaluate()}")
        print()
        if unique_move:
            pg.time.wait(500)
        self.board_update()
        if not self.winner and self.turn == self.ai_player.color:
            self.ai_play_turn()
        # Leave selection disabled on a decided board so a click cannot act on a
        # stale move set once the game is over.
        self.selection_activated = not self.winner
        # Clicks made while the search blocked the loop are not moves the player
        # chose in this position, so they are discarded rather than replayed.
        pg.event.clear(pg.MOUSEBUTTONDOWN)

    def ai_play_turn_full(self) -> None:
        """
        Plays a full turn for the AI while also printing the evaluation
        of the position before the AI starts thinking.
        """
        print(f"Current evaluation: {self.evaluate()}")
        print()
        self.ai_play_turn()
        print("--------------------------------------------------")
        print()
