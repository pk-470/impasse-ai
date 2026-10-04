"""
Game-logic regression tests for impasse.position.

Split into two groups:
  * invariants  - encode behaviour that is already correct and must never change
                  (these also guard the later performance refactors);
  * bug fixes   - encode the *correct* behaviour for the crowning/turn defects;
                  red on the unfixed engine, green once the fixes land.
"""

import random

import pytest

from impasse.position import (
    BLACK,
    INITIAL_STATE,
    WHITE,
    WIN_VALUE,
    Cell,
    Piece,
    Position,
    _make_state_hash,
)
from tests.boards import make_board


def perft(pos: Position, depth: int) -> int:
    """Count move sequences of length `depth` from `pos` (turn changes included)."""
    if depth == 0 or pos.winner is not None:
        return 1
    total = 0
    for origin, targets in pos.all_legal_moves.items():
        for target, tag in targets.items():
            total += perft(pos.new_position_after_move(origin, target, tag), depth - 1)
    return total


def _empty_board() -> dict[Cell, Piece | None]:
    """An empty {cell: piece} mapping, to be filled and passed to make_board."""
    return {}


# --------------------------------------------------------------------------- #
# Invariants (green on the current engine; guard the perf refactors)
# --------------------------------------------------------------------------- #


def test_start_move_counts():
    p = Position()
    assert len(p.all_legal_moves) == 4
    assert sum(len(t) for t in p.all_legal_moves.values()) == 22


@pytest.mark.parametrize("depth,expected", [(1, 22), (2, 492), (3, 9692)])
def test_perft_start(depth, expected):
    assert perft(Position(), depth) == expected


def test_start_eval_is_symmetric():
    assert Position().evaluate() == 0


def test_evaluate_is_turn_independent():
    # evaluate() is a pure function of the board; the same board must score the
    # same regardless of side to move (this guards the eval-memoization refactor).
    pw = Position(state=INITIAL_STATE.copy(), turn=WHITE)
    pb = Position(state=INITIAL_STATE.copy(), turn=BLACK)
    assert pw.evaluate() == pb.evaluate()


def test_incremental_zobrist_matches_from_scratch():
    random.seed(7)
    pos = Position()
    for _ in range(60):
        if pos.winner is not None:
            break
        origin = random.choice(list(pos.all_legal_moves))
        target = random.choice(list(pos.all_legal_moves[origin]))
        pos = pos.new_position_after_move(
            origin, target, pos.all_legal_moves[origin][target]
        )
        assert pos.state_hash == _make_state_hash(pos.state)


def test_new_position_does_not_mutate_source():
    p = Position()
    before_state, before_hash = p.state.copy(), p.state_hash
    origin = next(iter(p.all_legal_moves))
    target = next(iter(p.all_legal_moves[origin]))
    p.new_position_after_move(origin, target, p.all_legal_moves[origin][target])
    assert p.state == before_state
    assert p.state_hash == before_hash


def test_piece_at_decodes_pieces():
    # piece_at is the accessor the GUI renders from; it must report each piece as a
    # (color, type) pair and None for empty cells, whatever the internal encoding.
    p = Position()
    assert p.piece_at((0, 0)) == (WHITE, 1)
    assert p.piece_at((1, 7)) == (WHITE, 2)
    assert p.piece_at((0, 6)) == (BLACK, 1)
    assert p.piece_at((1, 1)) == (BLACK, 2)
    assert p.piece_at((3, 3)) is None


def test_winner_on_last_bear_off():
    state = _empty_board()
    state[(7, 7)] = (WHITE, 1)  # cornered single -> impasse -> forced bear-off
    state[(0, 6)] = (BLACK, 1)  # black still on the board
    pos = Position(state=make_board(state), turn=WHITE)
    assert pos.all_legal_moves == {(7, 7): {None: "B"}}
    nxt = pos.new_position_after_move((7, 7), None, "B")
    assert nxt.checkers_total[WHITE] == 0
    assert nxt.winner == WHITE


# --------------------------------------------------------------------------- #
# Bug fixes (red on the unfixed engine)
# --------------------------------------------------------------------------- #


def test_chained_crowning_keeps_turn():
    """
    Bug #3 (update 'C'): crowning one of several mandatory crownings must keep
    the turn with the mover and offer the remaining crownings."""
    state = _empty_board()
    state.update(
        {
            (1, 7): (WHITE, 1),  # on BLACK home row -> must be crowned
            (3, 7): (WHITE, 1),
            (5, 7): (WHITE, 1),
            (0, 0): (WHITE, 1),  # helper single
            (7, 7): (BLACK, 1),
        }
    )
    pos = Position(state=make_board(state), turn=WHITE)
    tags = {tag for targets in pos.all_legal_moves.values() for tag in targets.values()}
    assert tags == {"C"}, "all moves should be forced crownings"

    nxt = pos.new_position_after_move((0, 0), (5, 7), "C")
    assert nxt.turn == WHITE, (
        "turn must stay with the mover while a crowning is still owed"
    )
    nxt_tags = {
        tag for targets in nxt.all_legal_moves.values() for tag in targets.values()
    }
    assert nxt_tags == {"C"}, "the remaining mandatory crownings must be offered"


def test_incoming_mandatory_crowning_is_offered():
    """
    Bug #4 (change_turn): when the turn passes to a side that owes a mandatory
    crowning, change_turn must offer it (get_all_legal_moves, not get_other_moves)."""
    state = _empty_board()
    state.update(
        {
            (3, 7): (WHITE, 1),  # white single parked on BLACK's home row
            (0, 0): (WHITE, 1),  # helper single -> crowning is mandatory for white
            (5, 5): (BLACK, 1),
        }
    )
    pos = Position(state=make_board(state), turn=BLACK)
    assert pos.all_legal_moves[(5, 5)][(6, 4)] == "S"
    nxt = pos.new_position_after_move((5, 5), (6, 4), "S")
    assert nxt.turn == WHITE
    tags = {tag for targets in nxt.all_legal_moves.values() for tag in targets.values()}
    assert "C" in tags, "the incoming mandatory crowning must be in the legal moves"


def test_crownings_offer_every_legal_helper_target_pair():
    """
    Every (helper, furthest-row single) pair is offered, not just the last one.

    Keying the crownings map by helper origin alone used to drop A1 -> B8 here.
    """
    board = _empty_board()
    board[(1, 7)] = (WHITE, 1)
    board[(3, 7)] = (WHITE, 1)
    board[(0, 0)] = (WHITE, 1)
    board[(6, 0)] = (BLACK, 2)
    pos = Position(state=make_board(board), turn=WHITE)
    offered = {
        (origin, target)
        for origin, targets in pos.all_legal_moves.items()
        for target, tag in targets.items()
        if tag == "C"
    }
    assert offered == {
        ((0, 0), (1, 7)),
        ((0, 0), (3, 7)),
        ((1, 7), (3, 7)),
        ((3, 7), (1, 7)),
    }


def test_transpose_bear_off_checks_for_crownings():
    """'TB' creates a new single, which can owe a crowning, so the turn must stay."""
    board = _empty_board()
    board[(1, 7)] = (WHITE, 1)  # uncrowned single on White's furthest row
    board[(2, 0)] = (WHITE, 1)  # single in White's nearest row
    board[(1, 1)] = (WHITE, 2)  # double that can transpose onto it
    board[(6, 0)] = (BLACK, 2)
    pos = Position(state=make_board(board), turn=WHITE)
    update = pos.apply_move((1, 1), (2, 0), "TB")
    pos.update(update, "TB")
    assert pos.turn == WHITE, "a crowning is owed, so the turn must not change"
    assert all(tag == "C" for t in pos.all_legal_moves.values() for tag in t.values())


def test_winner_is_read_off_an_explicitly_given_board():
    """A board with no White checkers is a White win, without being told so."""
    board = _empty_board()
    board[(0, 6)] = (BLACK, 1)
    board[(7, 7)] = (BLACK, 2)
    pos = Position(state=make_board(board), turn=BLACK)
    assert pos.winner == WHITE
    assert pos.evaluate() == WIN_VALUE
    # None still means "nobody has won", so copies of a live game are unaffected.
    assert Position(state=make_board(board), turn=BLACK, winner=None).winner is None
