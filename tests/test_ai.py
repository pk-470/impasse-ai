"""
Search regression tests for impasse.ai.

One class per method under test. Several encode the correct behaviour for
defects in the search, and were red on the unfixed engine.
"""

from math import inf

import pytest

import impasse.ai as ai_mod
from impasse.ai import AI, INFINITY
from impasse.position import BLACK, INITIAL_STATE, WHITE, Cell, Move, Piece, Position
from tests.boards import make_board


def _armed(color: tuple[int, int, int] = WHITE) -> AI:
    """An AI whose search state is set so alpha_beta can be called directly."""
    ai = AI(color)
    ai.search_start_time = 0
    ai.min_search_depth_reached = False
    ai.completed_any_depth = True
    return ai


def _legal(position: Position, move: Move) -> bool:
    """Whether `move` is one of `position`'s legal moves, tag included."""
    origin, target, tag = move
    return (
        origin in position.all_legal_moves
        and target in position.all_legal_moves[origin]
        and position.all_legal_moves[origin][target] == tag
    )


def _tt_value(ai: AI, pos: Position) -> int:
    """The stored value for a position, asserting there is an entry at all."""
    entry = ai.tt_retrieve(pos)
    assert entry is not None
    return entry[0]


def _forced_result_position() -> Position:
    """A sparse position whose game tree resolves to a forced win for White."""
    return Position(
        state=make_board(
            {
                (0, 0): (WHITE, 1),
                (2, 0): (WHITE, 2),
                (7, 7): (BLACK, 1),
                (5, 7): (BLACK, 2),
            }
        ),
        turn=WHITE,
    )


@pytest.fixture
def no_time_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable both time cutoffs, so fixed-depth alpha_beta calls are deterministic."""
    monkeypatch.setattr(ai_mod, "MILLISECONDS_PER_MOVE", inf)
    monkeypatch.setattr(ai_mod, "MAX_MILLISECONDS_PER_MOVE", inf)


class TestAlphaBeta:
    def test_returns_a_legal_move(self, no_time_limit: None) -> None:
        pos = Position()
        _, move = _armed().alpha_beta(pos, 3, -INFINITY, INFINITY)
        assert move is not None
        assert _legal(pos, move)

    def test_warm_caches_are_populated_and_give_the_same_result(
        self, no_time_limit: None
    ) -> None:
        pos = Position()
        ai = _armed()
        first = ai.alpha_beta(pos, 4, -INFINITY, INFINITY)
        assert ai.eval_cache, "leaf evaluations should be memoized"
        assert ai.transposition_table, "search results should be stored"
        assert ai.alpha_beta(pos, 4, -INFINITY, INFINITY) == first

    def test_a_tt_primed_for_the_other_side_does_not_change_the_result(
        self, no_time_limit: None
    ) -> None:
        """The key carries the side to move, so the two searches cannot collide."""
        ai = _armed()
        ai.alpha_beta(
            Position(state=INITIAL_STATE.copy(), turn=WHITE), 3, -INFINITY, INFINITY
        )
        primed = ai.alpha_beta(
            Position(state=INITIAL_STATE.copy(), turn=BLACK), 2, -INFINITY, INFINITY
        )
        fresh = _armed().alpha_beta(
            Position(state=INITIAL_STATE.copy(), turn=BLACK), 2, -INFINITY, INFINITY
        )
        assert primed == fresh


class TestTranspositionTable:
    @pytest.mark.parametrize(
        "stores,expected",
        [
            ([(5, 2), (7, 6)], 7),  # the deeper result is kept
            ([(5, 2), (7, 6), (9, 3)], 7),  # a shallower one does not displace it
            ([(5, 2), (7, 6), (11, 6)], 11),  # an equal-or-deeper one replaces
        ],
    )
    def test_store_keeps_the_more_deeply_searched_entry(
        self, stores: list[tuple[int, int]], expected: int
    ) -> None:
        ai = AI(WHITE)
        pos = Position()
        for value, depth in stores:
            ai.tt_store(pos, value, None, "E", depth)
        assert _tt_value(ai, pos) == expected

    def test_eviction_bounds_both_tables_without_corrupting_the_value(
        self, no_time_limit: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pos = Position()
        monkeypatch.setattr(ai_mod, "TT_MAX_ENTRIES", 50)
        monkeypatch.setattr(ai_mod, "EVAL_CACHE_MAX_ENTRIES", 50)
        bounded = _armed()
        bounded_value, _ = bounded.alpha_beta(pos, 4, -INFINITY, INFINITY)
        assert len(bounded.transposition_table) <= 50
        assert len(bounded.eval_cache) <= 50

        monkeypatch.setattr(ai_mod, "TT_MAX_ENTRIES", 10_000_000)
        monkeypatch.setattr(ai_mod, "EVAL_CACHE_MAX_ENTRIES", 10_000_000)
        unbounded_value, _ = _armed().alpha_beta(pos, 4, -INFINITY, INFINITY)
        assert bounded_value == unbounded_value


class TestIterativeDeepening:
    def test_a_finite_budget_still_returns_a_legal_move(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ai_mod, "MIN_SEARCH_DEPTH", 1)
        monkeypatch.setattr(ai_mod, "MILLISECONDS_PER_MOVE", 50)
        monkeypatch.setattr(ai_mod, "MAX_MILLISECONDS_PER_MOVE", 100)
        pos = Position()
        depth, _, move = AI(WHITE).iterative_deepening(pos)
        assert depth >= 1
        assert move is not None
        assert _legal(pos, move)

    def test_survives_a_timeout_before_the_first_depth_completes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ai_mod, "MAX_MILLISECONDS_PER_MOVE", -1)
        pos = Position()
        _, _, best_move = AI(WHITE).iterative_deepening(pos)
        assert best_move is not None
        assert _legal(pos, best_move)

    def test_stops_on_a_forced_result(self, no_time_limit: None) -> None:
        depth, value, move = AI(WHITE).iterative_deepening(_forced_result_position())
        assert abs(value) >= ai_mod.MATE_THRESHOLD, (
            "a forced result is a mate-magnitude value"
        )
        assert depth < ai_mod.MAX_SEARCH_DEPTH, (
            "stopped on the proven result, not the cap"
        )
        assert move is not None

    def test_respects_the_depth_cap(
        self, no_time_limit: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ai_mod, "MAX_SEARCH_DEPTH", 8)
        depth, _, move = AI(WHITE).iterative_deepening(_forced_result_position())
        assert depth == 8, "the cap stops the search; mate is deeper than 8 here"
        assert move is not None


class TestSuggestedMove:
    def test_a_single_legal_move_is_returned_without_searching(self) -> None:
        pos = Position(
            state=make_board({(7, 7): (WHITE, 1), (0, 6): (BLACK, 1)}), turn=WHITE
        )
        assert pos.all_legal_moves == {(7, 7): {None: "B"}}
        origin, target, tag, unique = AI(WHITE).suggested_move(pos)
        assert (origin, target, tag) == ((7, 7), None, "B")
        assert unique is True


class TestDevStats:
    def test_counters_are_consistent_when_dev_is_on(
        self, no_time_limit: None
    ) -> None:
        ai = _armed()
        ai.dev = True
        ai.reset_stats()
        ai.alpha_beta(Position(), 4, -INFINITY, INFINITY)
        assert ai.nodes > 0
        assert ai.tt_stores > 0
        # Hits are a subset of lookups; cutoffs are a subset of hits.
        assert ai.tt_lookups >= ai.tt_hits >= ai.tt_cutoffs >= 0
        assert ai.eval_lookups >= ai.eval_hits >= 0

    def test_counters_stay_zero_when_dev_is_off(self, no_time_limit: None) -> None:
        ai = _armed()
        ai.alpha_beta(Position(), 4, -INFINITY, INFINITY)
        assert (
            ai.nodes,
            ai.tt_lookups,
            ai.tt_hits,
            ai.tt_stores,
            ai.eval_lookups,
        ) == (0, 0, 0, 0, 0)

    def test_collecting_them_does_not_change_the_search(
        self, no_time_limit: None
    ) -> None:
        pos = Position()
        instrumented = _armed()
        instrumented.dev = True
        assert instrumented.alpha_beta(
            pos, 4, -INFINITY, INFINITY
        ) == _armed().alpha_beta(pos, 4, -INFINITY, INFINITY)


class TestWinValue:
    def test_a_real_win_outranks_a_heuristic_lead(self) -> None:
        won = Position(
            state=make_board({(7, 7): (WHITE, 1), (0, 6): (BLACK, 1)}), turn=WHITE
        ).new_position_after_move((7, 7), None, "B")
        assert won.winner == WHITE

        ahead: dict[Cell, Piece | None] = {
            (0, 0): (WHITE, 1),
            (2, 0): (WHITE, 1),
        }
        for cell in ((1, 1), (3, 1), (5, 1), (7, 1), (1, 7), (3, 7), (5, 7), (7, 7)):
            ahead[cell] = (BLACK, 1)
        heuristic = Position(state=make_board(ahead), turn=WHITE)
        assert heuristic.winner is None
        assert won.evaluate() > heuristic.evaluate()
