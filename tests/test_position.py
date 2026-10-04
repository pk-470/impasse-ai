"""
Game-logic regression tests for impasse.position.

One class per method under test. Several encode the correct behaviour for
crowning and turn-flow defects, and were red on the unfixed engine.
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


class TestMoveGeneration:
    def test_start_move_counts(self) -> None:
        p = Position()
        assert len(p.all_legal_moves) == 4
        assert sum(len(t) for t in p.all_legal_moves.values()) == 22

    @pytest.mark.parametrize("depth,expected", [(1, 22), (2, 492), (3, 9692)])
    def test_perft_from_the_start(self, depth: int, expected: int) -> None:
        assert perft(Position(), depth) == expected


class TestEvaluate:
    def test_the_start_position_is_symmetric(self) -> None:
        assert Position().evaluate() == 0

    def test_is_turn_independent(self) -> None:
        pw = Position(state=INITIAL_STATE.copy(), turn=WHITE)
        pb = Position(state=INITIAL_STATE.copy(), turn=BLACK)
        assert pw.evaluate() == pb.evaluate()


class TestPieceAt:
    @pytest.mark.parametrize(
        "cell,piece",
        [
            ((0, 0), (WHITE, 1)),
            ((1, 7), (WHITE, 2)),
            ((0, 6), (BLACK, 1)),
            ((1, 1), (BLACK, 2)),
            ((3, 3), None),
        ],
    )
    def test_decodes_each_square_to_a_colour_and_type(
        self, cell: Cell, piece: Piece | None
    ) -> None:
        assert Position().piece_at(cell) == piece


class TestStateHash:
    def test_incremental_matches_from_scratch(self) -> None:
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


class TestNewPositionAfterMove:
    def test_does_not_mutate_the_source(self) -> None:
        p = Position()
        before_state, before_hash = p.state.copy(), p.state_hash
        origin = next(iter(p.all_legal_moves))
        target = next(iter(p.all_legal_moves[origin]))
        p.new_position_after_move(origin, target, p.all_legal_moves[origin][target])
        assert p.state == before_state
        assert p.state_hash == before_hash


class TestWinner:
    def test_is_set_on_the_last_bear_off(self) -> None:
        pos = Position(
            state=make_board(
                {
                    (7, 7): (WHITE, 1),  # cornered single -> forced bear-off
                    (0, 6): (BLACK, 1),
                }
            ),
            turn=WHITE,
        )
        assert pos.all_legal_moves == {(7, 7): {None: "B"}}
        nxt = pos.new_position_after_move((7, 7), None, "B")
        assert nxt.checkers_total[WHITE] == 0
        assert nxt.winner == WHITE

    def test_is_read_off_an_explicitly_given_board(self) -> None:
        board = make_board({(0, 6): (BLACK, 1), (7, 7): (BLACK, 2)})
        assert Position(state=board.copy(), turn=BLACK).winner == WHITE
        assert Position(state=board.copy(), turn=BLACK).evaluate() == WIN_VALUE
        # None still means "nobody has won".
        assert Position(state=board, turn=BLACK, winner=None).winner is None


class TestCrownings:
    def test_a_chained_crowning_keeps_the_turn(self) -> None:
        pos = Position(
            state=make_board(
                {
                    (1, 7): (WHITE, 1),  # on BLACK's home row -> must be crowned
                    (3, 7): (WHITE, 1),
                    (5, 7): (WHITE, 1),
                    (0, 0): (WHITE, 1),  # helper single
                    (7, 7): (BLACK, 1),
                }
            ),
            turn=WHITE,
        )
        tags = {
            tag for targets in pos.all_legal_moves.values() for tag in targets.values()
        }
        assert tags == {"C"}, "all moves should be forced crownings"

        nxt = pos.new_position_after_move((0, 0), (5, 7), "C")
        assert nxt.turn == WHITE, (
            "turn must stay with the mover while a crowning is still owed"
        )
        nxt_tags = {
            tag for targets in nxt.all_legal_moves.values() for tag in targets.values()
        }
        assert nxt_tags == {"C"}, "the remaining mandatory crownings must be offered"

    def test_an_incoming_mandatory_crowning_is_offered(self) -> None:
        pos = Position(
            state=make_board(
                {
                    (3, 7): (WHITE, 1),  # white single parked on BLACK's home row
                    (0, 0): (WHITE, 1),  # helper -> the crowning is mandatory
                    (5, 5): (BLACK, 1),
                }
            ),
            turn=BLACK,
        )
        assert pos.all_legal_moves[(5, 5)][(6, 4)] == "S"
        nxt = pos.new_position_after_move((5, 5), (6, 4), "S")
        assert nxt.turn == WHITE
        tags = {
            tag for targets in nxt.all_legal_moves.values() for tag in targets.values()
        }
        assert "C" in tags, "the incoming mandatory crowning must be in the legal moves"

    def test_every_legal_helper_target_pair_is_offered(self) -> None:
        pos = Position(
            state=make_board(
                {
                    (1, 7): (WHITE, 1),
                    (3, 7): (WHITE, 1),
                    (0, 0): (WHITE, 1),
                    (6, 0): (BLACK, 2),
                }
            ),
            turn=WHITE,
        )
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

    def test_a_transpose_bear_off_keeps_the_turn(self) -> None:
        """A "TB" leaves a new single on the origin, which can itself owe a crowning."""
        pos = Position(
            state=make_board(
                {
                    (1, 7): (WHITE, 1),  # uncrowned single on White's furthest row
                    (2, 0): (WHITE, 1),  # single in White's nearest row
                    (1, 1): (WHITE, 2),  # double that can transpose onto it
                    (6, 0): (BLACK, 2),
                }
            ),
            turn=WHITE,
        )
        pos.update(pos.apply_move((1, 1), (2, 0), "TB"), "TB")
        assert pos.turn == WHITE, "a crowning is owed, so the turn must not change"
        assert all(
            tag == "C" for t in pos.all_legal_moves.values() for tag in t.values()
        )
