"""
Alpha-beta search for Impasse.

Fixed-depth alpha-beta with move ordering, iterative deepening under a time
budget, a transposition table and a leaf-evaluation cache. The search works in
ints throughout, so a compiled build never boxes a float into a table entry.
"""

import time
from collections.abc import Callable
from typing import Final, cast

from impasse.position import (
    BLACK,
    COLOR_CODES,
    DIAG_RAYS,
    EMPTY,
    OPPOSITE_COLOR,
    SINGLE_CODE,
    TURN_BIT,
    WHITE,
    WIN_VALUE,
    Cell,
    Color,
    MoveTag,
    Position,
)

Move = tuple[Cell, Cell | None, MoveTag]

__all__ = [
    "AI",
    "DOUBLE_RAYS",
    "EVAL_CACHE_MAX_ENTRIES",
    "INFINITY",
    "MATE_THRESHOLD",
    "MAX_MILLISECONDS_PER_MOVE",
    "MAX_SEARCH_DEPTH",
    "MILLISECONDS_PER_MOVE",
    "MIN_SEARCH_DEPTH",
    "SINGLE_RAYS",
    "TICK_INTERVAL_NODES",
    "TIME_CHECK_INTERVAL_NODES",
    "TT_MAX_ENTRIES",
    "ABTimeOut",
    "Move",
    "TTEntry",
    "milliseconds",
]

# The DIAG_RAYS rows for each colour.
DOUBLE_RAYS: Final[dict[Color, list[list[int]]]] = {
    color: DIAG_RAYS[(color, 2)] for color in (WHITE, BLACK)
}
SINGLE_RAYS: Final[dict[Color, list[list[int]]]] = {
    color: DIAG_RAYS[(color, 1)] for color in (WHITE, BLACK)
}
# (value, best move, bound flag, depth searched)
TTEntry = tuple[int, Move | None, str, int]

# Not Final: reassigned by the tests and scripts/bench.py.
MIN_SEARCH_DEPTH = 5
MILLISECONDS_PER_MOVE: float = 6000
MAX_MILLISECONDS_PER_MOVE: float = 10000

# Backstop so a cheaply-resolved position cannot spin the depth counter.
MAX_SEARCH_DEPTH = 64
# A value this large is a proven win or loss, so deepening can stop.
MATE_THRESHOLD: Final = WIN_VALUE // 2
# The initial search window, clear of WIN_VALUE.
INFINITY: Final = WIN_VALUE * 2

TICK_INTERVAL_NODES: Final = 4096
TIME_CHECK_INTERVAL_NODES: Final = 512

TT_MAX_ENTRIES = 2_000_000
EVAL_CACHE_MAX_ENTRIES = 2_000_000


class ABTimeOut(Exception):
    """Raised to abandon a search that has run out of time."""


def milliseconds(seconds: float) -> int:
    """Convert a time in seconds to whole milliseconds."""
    return int(seconds * 1000)


class AI:
    """
    A class to take care of the logic behind an AI player. Uses alpha-beta search
    with move-ordering, iterative deepening and a transposition table.
    """

    def __init__(self, color: Color, dev: bool = False) -> None:
        """
        Create an AI player.

        Args:
            color: The colour this AI plays.
            dev: When True, per-move search diagnostics (transposition-table visits,
                eval-cache hits, node counts and speed) are collected and printed.
        """
        self.color: Color = color
        self.dev: bool = dev
        # Keyed by hash * 2 + turn bit.
        self.transposition_table: dict[int, TTEntry] = {}
        self.search_start_time: int = 0
        self.min_search_depth_reached: bool = False
        self.completed_any_depth: bool = False
        # evaluate() is turn-independent, so leaves are keyed on the board alone.
        self.eval_cache: dict[int, int] = {}
        self.killers: list[list[Move | None]] = [
            [None, None] for _ in range(MAX_SEARCH_DEPTH + 2)
        ]
        self.history: dict[Move, int] = {}
        self.on_tick: Callable[[], None] | None = None
        self.current_depth: int = 0
        self._tick_countdown: int = TICK_INTERVAL_NODES
        self._time_countdown: int = TIME_CHECK_INTERVAL_NODES
        self.reset_stats()

    def reset_stats(self) -> None:
        """Zero the per-move search diagnostic counters."""
        self.nodes = 0
        self.tt_lookups = 0
        self.tt_hits = 0
        self.tt_cutoffs = 0
        self.tt_stores = 0
        self.eval_lookups = 0
        self.eval_hits = 0

    def tt_retrieve(self, position: Position) -> TTEntry | None:
        """
        Look up a position's transposition-table entry.

        Args:
            position: The position to look up.

        Returns:
            The stored (value, move, flag, depth) entry, or None if the position
            has not been searched. Returning the entry itself rather than a
            filled-in sentinel keeps a miss -- the common case -- from building
            a tuple only to be thrown away.
        """
        entry = self.transposition_table.get(
            position.state_hash * 2 + TURN_BIT[position.turn]
        )
        if self.dev:
            self.tt_lookups += 1
            if entry is not None:
                self.tt_hits += 1
        return entry

    def tt_store(
        self,
        position: Position,
        value: int,
        move: Move | None,
        flag: str,
        depth: int,
    ) -> None:
        """
        Record a search result for a position, keeping the more reliable entry.

        A result searched to a greater depth is kept in preference to a shallower
        one for the same position; the table is bounded in size, evicting the
        oldest entries first.

        Args:
            position: The position the result is for.
            value: The alpha-beta value found.
            move: The best move found, if any.
            flag: The bound type, one of "E" (exact), "L" (lower), "U" (upper).
            depth: The search depth the result was computed at.
        """
        if self.dev:
            self.tt_stores += 1
        key = position.state_hash * 2 + TURN_BIT[position.turn]
        table = self.transposition_table
        existing = table.get(key)
        if existing is not None:
            # Keep the deeper entry; on a tie the newer result wins.
            if existing[3] > depth:
                return
        elif len(table) >= TT_MAX_ENTRIES:
            del table[next(iter(table))]
        table[key] = (value, move, flag, depth)

    def ordered_moves(
        self,
        position: Position,
        most_promising_move: Move | None = None,
        depth: int = 0,
    ) -> list[Move]:
        """
        Order a position's legal moves to improve alpha-beta pruning.

        Args:
            position: The position whose moves to order.
            most_promising_move: A move to try first ahead of all others, if any.
            depth: The remaining search depth, selecting the killer-move slot.

        Returns:
            The legal moves as (origin, target, tag) tuples, most promising first:
            the most_promising_move, then killer moves, forced crownings, bear offs,
            slides that block enemy doubles then singles (two blocks before one),
            potential crownings, transposes, and finally the remaining slides, most
            cutoffs first (longest first until the history table has entries).
        """
        check_first: list[Move] = []
        killer_moves: list[Move] = []
        killer_1, killer_2 = self.killers[depth]
        bear_offs: list[Move] = []
        crownings: list[Move] = []
        potential_crownings: list[Move] = []
        transposes: list[Move] = []
        slides_blocking_doubles_once: list[Move] = []
        slides_blocking_doubles_twice: list[Move] = []
        slides_blocking_singles_once: list[Move] = []
        slides_blocking_singles_twice: list[Move] = []
        other_slides: list[Move] = []
        state = position.state
        turn = position.turn
        enemy = OPPOSITE_COLOR[turn]
        enemy_double = COLOR_CODES[enemy][1]
        enemy_single = SINGLE_CODE[enemy]
        double_rays = DOUBLE_RAYS[turn]
        single_rays = SINGLE_RAYS[turn]
        seeded = most_promising_move is not None
        for move in position.moves_list:
            _, target, tag = move
            if seeded and move == most_promising_move:
                check_first = [move]
            # A killer is always tagged "S" or "T".
            elif tag == "S":
                if move == killer_1 or move == killer_2:
                    killer_moves.append(move)
                else:
                    # Rank by the enemy pieces the destination blocks, nearest first.
                    assert target is not None
                    base = (target[0] * 8 + target[1]) * 2
                    blocked_doubles = 0
                    for idx in double_rays[base]:
                        code = state[idx]
                        if code != EMPTY:
                            if code == enemy_double:
                                blocked_doubles = 1
                            break
                    for idx in double_rays[base + 1]:
                        code = state[idx]
                        if code != EMPTY:
                            if code == enemy_double:
                                blocked_doubles += 1
                            break
                    if blocked_doubles == 2:
                        slides_blocking_doubles_twice.append(move)
                    elif blocked_doubles:
                        slides_blocking_doubles_once.append(move)
                    else:
                        blocked_singles = 0
                        for idx in single_rays[base]:
                            code = state[idx]
                            if code != EMPTY:
                                if code == enemy_single:
                                    blocked_singles = 1
                                break
                        for idx in single_rays[base + 1]:
                            code = state[idx]
                            if code != EMPTY:
                                if code == enemy_single:
                                    blocked_singles += 1
                                break
                        if blocked_singles == 2:
                            slides_blocking_singles_twice.append(move)
                        elif blocked_singles:
                            slides_blocking_singles_once.append(move)
                        else:
                            other_slides.append(move)
            elif tag == "T":
                if move == killer_1 or move == killer_2:
                    killer_moves.append(move)
                else:
                    transposes.append(move)
            elif tag == "C":
                crownings.append(move)
            elif tag == "B" or tag == "SB" or tag == "TB":
                bear_offs.append(move)
            elif tag == "SC" or tag == "TC":
                potential_crownings.append(move)

        if len(other_slides) > 1:
            history = self.history
            if history:
                other_slides.sort(
                    reverse=True,
                    key=lambda m: (
                        history.get(m, 0),
                        abs(m[0][0] - cast(Cell, m[1])[0]),
                    ),
                )
            else:
                other_slides.sort(
                    reverse=True, key=lambda x: abs(x[0][0] - cast(Cell, x[1])[0])
                )
        ordered = check_first
        ordered.extend(killer_moves)
        ordered.extend(crownings)
        ordered.extend(bear_offs)
        ordered.extend(slides_blocking_doubles_twice)
        ordered.extend(slides_blocking_doubles_once)
        ordered.extend(slides_blocking_singles_twice)
        ordered.extend(slides_blocking_singles_once)
        ordered.extend(potential_crownings)
        ordered.extend(transposes)
        ordered.extend(other_slides)
        return ordered

    def alpha_beta(
        self,
        position: Position,
        depth: int,
        alpha: int,
        beta: int,
    ) -> tuple[int, Move | None]:
        """
        Search a position with alpha-beta to a fixed depth.

        Args:
            position: The position to search.
            depth: The number of full-move plies left to search.
            alpha: The lower bound on the value (best already assured to White).
            beta: The upper bound on the value (best already assured to Black).

        Returns:
            The (value, best_move) pair for the position; best_move is None at a
            leaf (depth 0 or a decided position).

        Raises:
            ABTimeOut: If the move's time budget is exceeded once at least one
                full depth has completed.
        """

        # Never before one depth has completed.
        self._time_countdown -= 1
        if self._time_countdown <= 0:
            self._time_countdown = TIME_CHECK_INTERVAL_NODES
            move_time = milliseconds(time.time()) - self.search_start_time
            if self.completed_any_depth and (
                (move_time > MILLISECONDS_PER_MOVE and self.min_search_depth_reached)
                or move_time > MAX_MILLISECONDS_PER_MOVE
            ):
                raise ABTimeOut

        if self.on_tick is not None:
            self._tick_countdown -= 1
            if self._tick_countdown <= 0:
                self._tick_countdown = TICK_INTERVAL_NODES
                self.on_tick()

        if self.dev:
            self.nodes += 1
        old_alpha, old_beta = alpha, beta
        tt_move: Move | None = None
        entry = self.tt_retrieve(position)
        if entry is not None:
            tt_value, tt_move, tt_flag, tt_depth = entry
            if tt_depth >= depth:
                if tt_flag == "E":
                    if self.dev:
                        self.tt_cutoffs += 1
                    return tt_value, tt_move
                elif tt_flag == "L":
                    alpha = max(alpha, tt_value)
                elif tt_flag == "U":
                    beta = min(beta, tt_value)
                if alpha >= beta:
                    if self.dev:
                        self.tt_cutoffs += 1
                    return tt_value, tt_move

        if position.winner or not depth:
            state_hash = position.state_hash
            cache = self.eval_cache
            cached = cache.get(state_hash)
            if self.dev:
                self.eval_lookups += 1
                if cached is not None:
                    self.eval_hits += 1
            if cached is None:
                cached = position.evaluate()
                if len(cache) >= EVAL_CACHE_MAX_ENTRIES:
                    del cache[next(iter(cache))]
                cache[state_hash] = cached
            return cached, None

        maximising = position.turn == WHITE
        value = -INFINITY if maximising else INFINITY
        best_move: Move | None = None
        for move in self.ordered_moves(position, tt_move, depth):
            origin, target, tag = move
            new_position = position.new_position_after_move(origin, target, tag)
            if new_position.turn == position.turn:
                local_value, _ = self.alpha_beta(new_position, depth, alpha, beta)
            else:
                local_value, _ = self.alpha_beta(new_position, depth - 1, alpha, beta)
            if maximising:
                if local_value > value:
                    value = local_value
                    best_move = move
                    alpha = max(alpha, value)
            elif local_value < value:
                value = local_value
                best_move = move
                beta = min(beta, value)
            if alpha >= beta:
                # Remember quiet cutoffs, to try them earlier elsewhere.
                if tag == "S" or tag == "T":
                    slot = self.killers[depth]
                    if slot[0] != move:
                        slot[1] = slot[0]
                        slot[0] = move
                    self.history[move] = self.history.get(move, 0) + depth * depth
                break

        # Bound type: "U" below the original window, "L" above it, "E" inside.
        if value <= old_alpha:
            flag = "U"
        elif value >= old_beta:
            flag = "L"
        else:
            flag = "E"
        self.tt_store(position, value, best_move, flag, depth)

        return value, best_move

    def iterative_deepening(self, position: Position) -> tuple[int, int, Move | None]:
        """
        Search a position to increasing depths within the move's time budget.

        Searches depth 1, 2, ...; once MIN_SEARCH_DEPTH is passed it keeps going
        until MILLISECONDS_PER_MOVE is spent, and stops unconditionally at
        MAX_MILLISECONDS_PER_MOVE. It also stops early once a forced win or loss is
        proven (its value cannot improve with depth) or MAX_SEARCH_DEPTH is reached.

        Args:
            position: The position to search.

        Returns:
            A (depth, value, best_move) triple from the deepest fully completed
            search; best_move is None only if the position has no legal move.
        """
        search_depth = 1
        self.min_search_depth_reached = False
        self.completed_any_depth = False
        if self.dev:
            self.reset_stats()
        self.search_start_time = milliseconds(time.time())
        self._tick_countdown = TICK_INTERVAL_NODES
        self._time_countdown = TIME_CHECK_INTERVAL_NODES
        # Fallback for a timeout before any depth completes (depth 1 always does).
        prev_search_depth: int = 0
        prev_value: int = position.evaluate()
        prev_best_move: Move | None = None
        while True:
            self.current_depth = search_depth
            if search_depth > MIN_SEARCH_DEPTH:
                self.min_search_depth_reached = True
            try:
                value, best_move = self.alpha_beta(
                    position,
                    search_depth,
                    -INFINITY,
                    INFINITY,
                )
            except ABTimeOut:
                break
            self.completed_any_depth = True
            prev_search_depth, prev_value, prev_best_move = (
                search_depth,
                value,
                best_move,
            )
            if abs(value) >= MATE_THRESHOLD or search_depth >= MAX_SEARCH_DEPTH:
                break
            search_depth += 1

        return prev_search_depth, prev_value, prev_best_move

    def print_dev_stats(self, depth: int) -> None:
        """
        Print the diagnostics gathered for the move just searched.

        Args:
            depth: The depth of the deepest completed search for the move.
        """
        elapsed_ms = milliseconds(time.time()) - self.search_start_time
        seconds = elapsed_ms / 1000
        nps = self.nodes / seconds if seconds else 0
        tt_hit_pct = 100 * self.tt_hits / self.tt_lookups if self.tt_lookups else 0
        eval_hit_pct = (
            100 * self.eval_hits / self.eval_lookups if self.eval_lookups else 0
        )
        print(
            f"[dev] depth {depth} | {elapsed_ms} ms | {self.nodes} nodes ({nps:,.0f}/s)"
        )
        print(
            f"[dev] TT: {self.tt_lookups} visits, {self.tt_hits} hits "
            f"({tt_hit_pct:.1f}%), {self.tt_cutoffs} cutoffs, {self.tt_stores} stores, "
            f"size {len(self.transposition_table)}"
        )
        print(
            f"[dev] eval cache: {self.eval_lookups} lookups, {self.eval_hits} hits "
            f"({eval_hit_pct:.1f}%), size {len(self.eval_cache)}"
        )

    def suggested_move(
        self, position: Position
    ) -> tuple[Cell | None, Cell | None, MoveTag | None, bool]:
        """
        Choose the best move for the player to move and print its evaluation.

        Args:
            position: The position to move in.

        Returns:
            An (origin, target, tag, unique_move) tuple: the chosen move and a
            flag that is True when it was the only legal move (returned without
            searching). The first three are None if there is no legal move.
        """
        moves = position.moves_list
        if len(moves) == 1:
            origin, target, tag = moves[0]
            single_value = position.evaluate()
            print(f"Alpha-Beta evaluation: {single_value} at depth 0 (one legal move)")
            if self.dev:
                print("[dev] 1 legal move — returned without searching")
            return origin, target, tag, True

        depth, value, best_move = self.iterative_deepening(position)
        if best_move is None:
            return None, None, None, False
        origin, target, tag = best_move
        print(f"Alpha-Beta evaluation: {value} at depth {depth}")
        if self.dev:
            self.print_dev_stats(depth)

        return origin, target, tag, False
