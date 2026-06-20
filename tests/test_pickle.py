"""Pickle round-trip lock for the board representation.

The GUI persists the raw board (self.state) to a recovery file via pickle and
reconstructs a position from it on undo. These tests pin that contract so the
flat int-board rewrite keeps the board picklable and faithfully reconstructable.
"""

import pickle
import random

from impasse.position import _make_state_hash, Position


def _midgame(seed=3, plies=25):
    """Return a position reached by `plies` seeded random moves."""
    rng = random.Random(seed)
    pos = Position()
    for _ in range(plies):
        if pos.winner is not None:
            break
        origin = rng.choice(list(pos.all_legal_moves))
        target = rng.choice(list(pos.all_legal_moves[origin]))
        pos = pos.new_position_after_move(origin, target, pos.all_legal_moves[origin][target])
    return pos


def _save_dict(pos):
    """Mirror the engine-relevant fields the GUI pickles for recovery."""
    return {
        "state": pos.state,
        "turn": pos.turn,
        "checkers_total": pos.checkers_total,
        "winner": pos.winner,
        "state_hash": pos.state_hash,
    }


def test_state_survives_pickle_roundtrip():
    """A pickled board reconstructs to an identical, consistent position."""
    pos = _midgame()
    data = pickle.loads(pickle.dumps(_save_dict(pos)))
    restored = Position(
        state=data["state"],
        turn=data["turn"],
        checkers_total=data["checkers_total"],
        winner=data["winner"],
        state_hash=data["state_hash"],
    )
    assert restored.state_hash == pos.state_hash
    assert restored.state_hash == _make_state_hash(restored.state)
    assert restored.checkers_total == pos.checkers_total
    assert restored.winner == pos.winner
    assert restored.evaluate() == pos.evaluate()
    assert restored.all_legal_moves == pos.all_legal_moves
