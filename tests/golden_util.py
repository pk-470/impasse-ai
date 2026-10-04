"""Representation-independent golden-master helpers for the performance refactor.

Everything here reads only the PUBLIC, cell-based API (all_legal_moves, evaluate,
checkers_total[WHITE/BLACK], winner, turn) and references the module WHITE/BLACK
constants rather than their raw values, so the digests stay identical across any
behaviour-preserving change to the internal board representation.
"""

import hashlib
import random
from math import inf

import impasse.ai as ai_mod
from impasse.ai import AI, INFINITY
from impasse.position import BLACK, WHITE, Position


def _canon_color(color):
    return 0 if color == WHITE else 1 if color == BLACK else -1


def _move_key(move):
    origin, target, tag = move
    return (origin[0], origin[1], (-1, -1) if target is None else target, tag)


def _legal_moves(pos):
    return sorted(
        ((o, t, tag) for o, ts in pos.all_legal_moves.items() for t, tag in ts.items()),
        key=_move_key,
    )


def position_features(pos):
    """A representation-independent tuple of everything observable about a position."""
    return (
        _canon_color(pos.turn),
        pos.evaluate(),
        pos.checkers_total[WHITE],
        pos.checkers_total[BLACK],
        _canon_color(pos.winner),
        _legal_moves(pos),
    )


def trajectory_digest(seed=12345, max_positions=4000, max_games=300):
    """Digest of the observable trajectory of seeded random self-play games.

    Deterministic given (seed) and the move generator; sensitive to any change in
    evaluate() or legal-move generation.
    """
    rng = random.Random(seed)
    h = hashlib.sha256()
    count = 0
    games = 0
    while count < max_positions and games < max_games:
        pos = Position()
        games += 1
        steps = 0
        while pos.winner is None and count < max_positions and steps < 500:
            moves = _legal_moves(pos)
            h.update(repr(position_features(pos)).encode())
            count += 1
            steps += 1
            origin, target, tag = moves[rng.randrange(len(moves))]
            pos = pos.new_position_after_move(origin, target, tag)
    return h.hexdigest(), count


def ordering_digest(seed=4242, max_positions=1200, max_games=200):
    """Digest of move *ordering*, which trajectory_digest cannot see.

    `_legal_moves` sorts, so the trajectory digest is blind both to the order
    move generation emits moves in and to the order `ai.ordered_moves` puts them
    in -- and that order decides which of several equally-valued moves the search
    returns, so it is behaviour. This locks both.

    The killer slots, the history table and the transposition-table move are
    synthesised from the seed rather than inherited from a real search, so the
    state the ordering depends on is fixed and both the history-sorted and the
    distance-sorted branch get exercised. The walk itself steps through sorted
    moves, so the positions sampled stay the same even if emission order changed
    -- otherwise a divergence would move the sample and hide itself.
    """
    rng = random.Random(seed)
    ai = AI(WHITE)
    h = hashlib.sha256()
    count = 0
    games = 0
    while count < max_positions and games < max_games:
        pos = Position()
        games += 1
        steps = 0
        while pos.winner is None and count < max_positions and steps < 500:
            moves = _legal_moves(pos)
            h.update(repr(pos.moves_list).encode())
            quiet = [m for m in moves if m[2] == "S" or m[2] == "T"]
            for depth in (1, 3):
                slot = ai.killers[depth]
                slot[0] = rng.choice(quiet) if quiet and rng.random() < 0.7 else None
                slot[1] = rng.choice(quiet) if quiet and rng.random() < 0.5 else None
            ai.history = {}
            if rng.random() < 0.6:
                for m in moves:
                    if rng.random() < 0.4:
                        ai.history[m] = rng.randrange(1, 400)
            for depth in (1, 3):
                for promising in (None, moves[rng.randrange(len(moves))]):
                    h.update(repr(ai.ordered_moves(pos, promising, depth)).encode())
            count += 1
            steps += 1
            pos = pos.new_position_after_move(*moves[rng.randrange(len(moves))])
    return h.hexdigest(), count


def _replay_to(seed, n_moves):
    rng = random.Random(seed)
    pos = Position()
    for _ in range(n_moves):
        if pos.winner is not None:
            break
        moves = _legal_moves(pos)
        origin, target, tag = moves[rng.randrange(len(moves))]
        pos = pos.new_position_after_move(origin, target, tag)
    return pos


def search_results(seed=999, offsets=(0, 6, 12, 20, 30), depth=4):
    """Fixed-depth alpha_beta (value, move) from several seeded midgame positions.

    Locks the exact search behaviour. Time cutoffs are disabled for determinism.
    """
    saved = (ai_mod.MILLISECONDS_PER_MOVE, ai_mod.MAX_MILLISECONDS_PER_MOVE)
    ai_mod.MILLISECONDS_PER_MOVE = inf
    ai_mod.MAX_MILLISECONDS_PER_MOVE = inf
    try:
        out = []
        for off in offsets:
            pos = _replay_to(seed, off)
            if pos.winner is not None:
                out.append((off, None, None))
                continue
            ai = AI(WHITE)
            ai.completed_any_depth = False
            value, move = ai.alpha_beta(pos, depth, -INFINITY, INFINITY)
            mv = None if move is None else (move[0], move[1], move[2])
            out.append((off, value, mv))
        return tuple(out)
    finally:
        ai_mod.MILLISECONDS_PER_MOVE, ai_mod.MAX_MILLISECONDS_PER_MOVE = saved


def _is_legal(pos, move):
    origin, target, tag = move
    return (
        origin in pos.all_legal_moves
        and target in pos.all_legal_moves[origin]
        and pos.all_legal_moves[origin][target] == tag
    )


def search_values_legal_tied(seed=999, offsets=(0, 6, 12, 20, 30), depth=4):
    """Per-offset (off, value, move_is_legal, move_attains_value) for the value-only
    golden.

    Tolerant of move-ordering changes (killers, history, aspiration, TT
    replacement) that can pick a different equally-good move: it pins the search
    value, and checks the chosen move is legal and attains that value (its child,
    searched independently with a full window, scores the same). Time cutoffs are
    disabled for determinism.
    """
    saved = (ai_mod.MILLISECONDS_PER_MOVE, ai_mod.MAX_MILLISECONDS_PER_MOVE)
    ai_mod.MILLISECONDS_PER_MOVE = inf
    ai_mod.MAX_MILLISECONDS_PER_MOVE = inf
    try:
        out = []
        for off in offsets:
            pos = _replay_to(seed, off)
            if pos.winner is not None:
                out.append((off, None, True, True))
                continue
            ai = AI(WHITE)
            ai.completed_any_depth = False
            value, move = ai.alpha_beta(pos, depth, -INFINITY, INFINITY)
            if move is None:
                out.append((off, value, False, False))
                continue
            legal = _is_legal(pos, move)
            child = pos.new_position_after_move(*move)
            child_depth = depth if child.turn == pos.turn else depth - 1
            verifier = AI(WHITE)
            verifier.completed_any_depth = False
            child_value, _ = verifier.alpha_beta(child, child_depth, -INFINITY, INFINITY)
            out.append((off, value, legal, child_value == value))
        return tuple(out)
    finally:
        ai_mod.MILLISECONDS_PER_MOVE, ai_mod.MAX_MILLISECONDS_PER_MOVE = saved


if __name__ == "__main__":
    digest, n = trajectory_digest()
    print(f"TRAJECTORY_DIGEST = {digest!r}  # over {n} positions")
    print(f"SEARCH_RESULTS = {search_results()!r}")
