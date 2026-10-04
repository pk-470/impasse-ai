"""Headless deterministic benchmark + profile for the Impasse engine.

Pure-Python builds only. The node and eval counters below work by monkeypatching
Position methods, and mypyc compiles intra-module calls to direct C calls that
bypass the patch -- so on a compiled build every count reads zero and cProfile
sees nothing, while the timings stay plausible. Rather than print that quietly,
refuse to run: `python scripts/build_native.py --clean` first.
"""

import cProfile
import io
import pstats
import sys
import time

import impasse.ai as ai_mod
from impasse.ai import AI
from impasse.position import WHITE, Position

if not ai_mod.__file__.endswith(".py"):
    sys.exit(
        "bench.py needs a pure-Python build; impasse.ai is compiled:\n"
        f"  {ai_mod.__file__}\n"
        "Its counters monkeypatch Position, which a compiled build bypasses, so"
        " every node/eval/nps column would read zero.\n"
        "Run `python scripts/build_native.py --clean` first."
    )

# Disable time-based cutoff so depths are deterministic.
ai_mod.MILLISECONDS_PER_MOVE = float("inf")
ai_mod.MAX_MILLISECONDS_PER_MOVE = float("inf")

NODES = 0
EVALS = 0

# Instrument node + eval counts.
_orig_new = Position.new_position_after_move


def _counted_new(self, *a, **k):
    global NODES
    NODES += 1
    return _orig_new(self, *a, **k)


Position.new_position_after_move = _counted_new

_orig_eval = Position.evaluate


def _counted_eval(self):
    global EVALS
    EVALS += 1
    return _orig_eval(self)


Position.evaluate = _counted_eval


def fresh_ai():
    ai = AI(WHITE)
    ai.search_start_time = ai_mod.milliseconds(time.time())
    ai.min_search_depth_reached = False
    return ai


def midgame_position(plies):
    """Play `plies` AI-vs-AI moves at shallow depth to reach a midgame state."""
    pos = Position()
    ai = fresh_ai()
    for _ in range(plies):
        if pos.winner:
            break
        _, mv = ai.alpha_beta(pos, 3, float("-inf"), float("inf"))
        if mv is None:
            break
        pos = pos.new_position_after_move(*mv)
    return pos


def bench(label, pos, depths):
    print(f"\n=== {label} (legal origins={len(pos.all_legal_moves)}) ===")
    for d in depths:
        global NODES, EVALS
        NODES = 0
        EVALS = 0
        ai = fresh_ai()
        t0 = time.perf_counter()
        val, mv = ai.alpha_beta(pos, d, float("-inf"), float("inf"))
        dt = time.perf_counter() - t0
        nps = NODES / dt if dt else 0
        print(
            f"  depth {d}: {dt:7.3f}s  nodes={NODES:>9}  evals={EVALS:>9}  "
            f"nps={nps:>10.0f}  val={val:.0f}  tt={len(ai.transposition_table)}"
        )


if __name__ == "__main__":
    open_pos = Position()
    bench("opening", open_pos, [1, 2, 3, 4, 5])

    mid = midgame_position(12)
    bench("midgame(~12 plies)", mid, [1, 2, 3, 4, 5])

    # Profile a representative depth from the opening.
    print("\n=== cProfile: opening depth 5 ===")
    NODES = 0
    EVALS = 0
    ai = fresh_ai()
    pr = cProfile.Profile()
    pr.enable()
    ai.alpha_beta(open_pos, 5, float("-inf"), float("inf"))
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(25)
    print(s.getvalue())
