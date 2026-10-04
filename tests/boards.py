"""
Readable board construction for the tests.

`Position` takes a flat list of 64 square codes. A test is clearer when it names
the pieces it places, so these build that list from a {cell: piece} mapping. The
engine itself has no dict-board form; this conversion belongs to the tests.
"""

from impasse.position import EMPTY, PIECE_TO_CODE, Cell, Piece, State


def make_board(pieces: dict[Cell, Piece | None]) -> State:
    """A flat board holding `pieces`, every other square empty."""
    flat = [EMPTY] * 64
    for cell, piece in pieces.items():
        flat[cell[0] * 8 + cell[1]] = PIECE_TO_CODE[piece]
    return flat
