"""Headless AI-vs-AI full game to completion at shallow depth: integration test
that the fixes don't crash through endgames / single-move positions / crownings."""

import impasse.ai as ai_mod
from impasse.ai import AI
from impasse.position import BLACK, WHITE, Position

# Bound search so the game runs fast but still uses iterative_deepening + TT.
ai_mod.MIN_SEARCH_DEPTH = 2
ai_mod.MILLISECONDS_PER_MOVE = 200
ai_mod.MAX_MILLISECONDS_PER_MOVE = 400

ais = {WHITE: AI(WHITE), BLACK: AI(BLACK)}
pos = Position()
moves = 0
crownings = 0
single_move_turns = 0
bear_offs = 0
import contextlib
import io

while pos.winner is None and moves < 2000:
    ai = ais[pos.turn]
    if (
        len(pos.all_legal_moves) == 1
        and len(next(iter(pos.all_legal_moves.values()))) == 1
    ):
        single_move_turns += 1
    with contextlib.redirect_stdout(io.StringIO()):
        origin, target, tag, unique = ai.suggested_move(pos)
    assert origin is not None and tag is not None, f"got null move at move {moves}"
    assert origin in pos.all_legal_moves and target in pos.all_legal_moves[origin], (
        f"illegal move {origin}->{target} ({tag}) at move {moves}"
    )
    if tag == "C":
        crownings += 1
    if tag in ("B", "SB", "TB"):
        bear_offs += 1
    pos = pos.new_position_after_move(origin, target, tag)
    moves += 1

print(
    f"game finished: {moves} moves, winner = "
    f"{'WHITE' if pos.winner == WHITE else 'BLACK' if pos.winner == BLACK else None}"
)
print(
    f"  crownings played: {crownings}, bear_offs: {bear_offs}, "
    f"single-move turns hit: {single_move_turns}"
)
print(f"  final checkers: W={pos.checkers_total[WHITE]} B={pos.checkers_total[BLACK]}")
print(
    f"  TT sizes: W={len(ais[WHITE].transposition_table)} B={len(ais[BLACK].transposition_table)}"
)
assert pos.winner is not None, "game did not finish"
print("OK: full game completed with no crash and all moves legal")
