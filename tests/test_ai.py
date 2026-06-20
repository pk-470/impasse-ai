"""Search/AI regression tests for impasse.ai.

Invariants guard the search; the bug-fix tests are red on the unfixed engine.
"""

from math import inf

import pytest

import impasse.ai as ai_mod
from impasse.ai import AI
from impasse.position import BLACK, INITIAL_STATE, WHITE, Position


def _armed(color=WHITE) -> AI:
    """An AI whose search-state attributes are set so alpha_beta can be called
    directly (independent of iterative_deepening)."""
    ai = AI(color)
    ai.search_start_time = 0
    ai.min_search_depth_reached = False
    ai.completed_any_depth = False
    return ai


def _legal(position: Position, move) -> bool:
    origin, target, tag = move
    return (
        origin in position.all_legal_moves
        and target in position.all_legal_moves[origin]
        and position.all_legal_moves[origin][target] == tag
    )


@pytest.fixture
def no_time_limit(monkeypatch):
    # Disable both time cutoffs so fixed-depth alpha_beta calls are deterministic.
    monkeypatch.setattr(ai_mod, "MILLISECONDS_PER_MOVE", inf)
    monkeypatch.setattr(ai_mod, "MAX_MILLISECONDS_PER_MOVE", inf)


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #


def test_alpha_beta_returns_a_legal_move(no_time_limit):
    pos = Position()
    _, move = _armed().alpha_beta(pos, 3, -inf, inf)
    assert move is not None
    assert _legal(pos, move)


def test_iterative_deepening_finite_budget_returns_legal_move(monkeypatch):
    """With a real (finite) time budget, iterative_deepening completes at least
    depth 1 and returns a legal move."""
    monkeypatch.setattr(ai_mod, "MIN_SEARCH_DEPTH", 1)
    monkeypatch.setattr(ai_mod, "MILLISECONDS_PER_MOVE", 50)
    monkeypatch.setattr(ai_mod, "MAX_MILLISECONDS_PER_MOVE", 100)
    pos = Position()
    depth, _, move = AI(WHITE).iterative_deepening(pos)
    assert depth >= 1
    assert move is not None
    assert _legal(pos, move)


def test_warm_caches_are_populated_and_consistent(no_time_limit):
    """A search fills the eval cache and transposition table, and re-searching the
    same position from those warm caches returns the identical (value, move)."""
    pos = Position()
    ai = _armed()
    ai.completed_any_depth = True
    first = ai.alpha_beta(pos, 4, -inf, inf)
    assert ai.eval_cache, "leaf evaluations should be memoized"
    assert ai.transposition_table, "search results should be stored"

    second = ai.alpha_beta(pos, 4, -inf, inf)
    assert second == first


# --------------------------------------------------------------------------- #
# Bug fixes (red on the unfixed engine)
# --------------------------------------------------------------------------- #


def test_suggested_move_single_legal_move_no_crash():
    """Bug #2: a position with exactly one legal move returns it (unique_move=True)
    via the fast path instead of raising UnboundLocalError."""
    state = {cell: None for cell in INITIAL_STATE}
    state[(7, 7)] = (WHITE, 1)  # impasse -> single forced bear-off
    state[(0, 6)] = (BLACK, 1)
    pos = Position(state=state, turn=WHITE)
    assert pos.all_legal_moves == {(7, 7): {None: "B"}}

    origin, target, tag, unique = AI(WHITE).suggested_move(pos)
    assert (origin, target, tag) == ((7, 7), None, "B")
    assert unique is True


def test_iterative_deepening_survives_immediate_timeout(monkeypatch):
    """Bug #5: if the very first (depth-1) search would time out, iterative_deepening
    must still return a legal move rather than raising UnboundLocalError."""
    monkeypatch.setattr(ai_mod, "MAX_MILLISECONDS_PER_MOVE", -1)
    pos = Position()
    depth, value, best_move = AI(WHITE).iterative_deepening(pos)
    assert best_move is not None
    assert _legal(pos, best_move)


def test_tt_depth_preferred_replacement_keeps_deeper_entry():
    """tt_store keeps the result searched to the greater depth for a position."""
    ai = AI(WHITE)
    pos = Position()
    ai.tt_store(pos, 5.0, None, "E", 2)
    ai.tt_store(pos, 7.0, None, "E", 6)
    assert ai.tt_retrieve(pos)[0] == 7.0  # deeper result kept
    ai.tt_store(pos, 9.0, None, "E", 3)  # shallower must not displace it
    assert ai.tt_retrieve(pos)[0] == 7.0
    ai.tt_store(pos, 11.0, None, "E", 6)  # equal-or-deeper replaces
    assert ai.tt_retrieve(pos)[0] == 11.0


def test_tt_and_eval_cache_eviction_is_bounded_and_sound(no_time_limit, monkeypatch):
    """A tiny TT/eval-cache cap bounds both tables yet still returns the correct
    search value: eviction only loses memoization, it never corrupts results."""
    monkeypatch.setattr(ai_mod, "TT_MAX_ENTRIES", 50)
    monkeypatch.setattr(ai_mod, "EVAL_CACHE_MAX_ENTRIES", 50)
    pos = Position()
    bounded = _armed()
    bounded.completed_any_depth = True
    bounded_value, _ = bounded.alpha_beta(pos, 4, -inf, inf)
    assert len(bounded.transposition_table) <= 50
    assert len(bounded.eval_cache) <= 50

    monkeypatch.setattr(ai_mod, "TT_MAX_ENTRIES", 10_000_000)
    monkeypatch.setattr(ai_mod, "EVAL_CACHE_MAX_ENTRIES", 10_000_000)
    unbounded = _armed()
    unbounded.completed_any_depth = True
    unbounded_value, _ = unbounded.alpha_beta(pos, 4, -inf, inf)
    assert bounded_value == unbounded_value


def test_tt_key_distinguishes_side_to_move(no_time_limit):
    """Bug #1: priming the transposition table from a White-to-move search must not
    corrupt a subsequent Black-to-move search of the same board."""
    ai = _armed()
    ai.completed_any_depth = True
    # Prime the TT with a White-to-move search of the opening.
    ai.alpha_beta(Position(state=INITIAL_STATE.copy(), turn=WHITE), 3, -inf, inf)
    primed = ai.alpha_beta(Position(state=INITIAL_STATE.copy(), turn=BLACK), 2, -inf, inf)

    fresh_ai = _armed()
    fresh_ai.completed_any_depth = True
    fresh = fresh_ai.alpha_beta(Position(state=INITIAL_STATE.copy(), turn=BLACK), 2, -inf, inf)

    assert primed == fresh, "TT primed for the other side must not change the result"


# --------------------------------------------------------------------------- #
# Dev-mode diagnostics
# --------------------------------------------------------------------------- #


def test_dev_stats_collected_and_consistent(no_time_limit):
    """With dev on, a search fills the diagnostic counters with consistent values."""
    pos = Position()
    ai = _armed()
    ai.dev = True
    ai.completed_any_depth = True
    ai.reset_stats()
    ai.alpha_beta(pos, 4, -inf, inf)
    assert ai.nodes > 0
    assert ai.tt_stores > 0
    # Hits are a subset of lookups; cutoffs are a subset of hits.
    assert ai.tt_lookups >= ai.tt_hits >= ai.tt_cutoffs >= 0
    assert ai.eval_lookups >= ai.eval_hits >= 0


def test_dev_counters_stay_zero_when_off(no_time_limit):
    """With dev off, the counters are never touched (zero search overhead)."""
    ai = _armed()
    ai.completed_any_depth = True
    ai.alpha_beta(Position(), 4, -inf, inf)
    assert (ai.nodes, ai.tt_lookups, ai.tt_hits, ai.tt_stores, ai.eval_lookups) == (
        0,
        0,
        0,
        0,
        0,
    )


def test_dev_instrumentation_does_not_change_search(no_time_limit):
    """Collecting diagnostics must not change the search value or move."""
    pos = Position()
    plain = _armed()
    plain.completed_any_depth = True
    instrumented = _armed()
    instrumented.dev = True
    instrumented.completed_any_depth = True
    assert instrumented.alpha_beta(pos, 4, -inf, inf) == plain.alpha_beta(pos, 4, -inf, inf)


# --------------------------------------------------------------------------- #
# Iterative-deepening termination (forced-result early exit + depth cap)
# --------------------------------------------------------------------------- #


def _forced_result_position() -> Position:
    """A sparse position whose game tree resolves to a forced win/loss for White."""
    state = {cell: None for cell in INITIAL_STATE}
    state[(0, 0)] = (WHITE, 1)
    state[(2, 0)] = (WHITE, 2)
    state[(7, 7)] = (BLACK, 1)
    state[(5, 7)] = (BLACK, 2)
    return Position(state=state, turn=WHITE)


def test_iterative_deepening_stops_on_forced_result(no_time_limit):
    """A proven forced win/loss stops iterative deepening via the mate exit, well
    below the depth cap, instead of spinning the depth counter."""
    depth, value, move = AI(WHITE).iterative_deepening(_forced_result_position())
    assert abs(value) >= ai_mod.MATE_THRESHOLD, "a forced result is a mate-magnitude value"
    assert depth < ai_mod.MAX_SEARCH_DEPTH, "stopped on the proven result, not the cap"
    assert move is not None


def test_iterative_deepening_respects_depth_cap(no_time_limit, monkeypatch):
    """When no forced result is yet in reach, the depth cap bounds the search rather
    than letting a cheaply-resolved position run away."""
    monkeypatch.setattr(ai_mod, "MAX_SEARCH_DEPTH", 8)
    depth, value, move = AI(WHITE).iterative_deepening(_forced_result_position())
    assert depth == 8, "the cap (mate is deeper than 8 here) is what stops the search"
    assert move is not None


def test_a_real_win_outranks_a_heuristic_line():
    """A terminal win must score above any heuristic position, so a winning move is
    never ranked below a merely material-heavy one."""
    win = {cell: None for cell in INITIAL_STATE}
    win[(7, 7)] = (WHITE, 1)  # White's only checker -> forced bear-off -> White wins
    win[(0, 6)] = (BLACK, 1)
    won = Position(state=win, turn=WHITE).new_position_after_move((7, 7), None, "B")
    assert won.winner == WHITE

    ahead = {cell: None for cell in INITIAL_STATE}
    ahead[(0, 0)] = (WHITE, 1)
    ahead[(2, 0)] = (WHITE, 1)  # White far ahead on material but not finished
    for cell in [(1, 1), (3, 1), (5, 1), (7, 1), (1, 7), (3, 7), (5, 7), (7, 7)]:
        ahead[cell] = (BLACK, 1)
    heuristic = Position(state=ahead, turn=WHITE)
    assert heuristic.winner is None
    assert won.evaluate() > heuristic.evaluate(), "an actual win must outscore a heuristic lead"
