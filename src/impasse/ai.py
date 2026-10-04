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

# DIAG_RAYS is keyed by (colour, checker type), so reading it per node hashes a
# tuple that contains a tuple. These are its two rows the move ordering wants,
# keyed by colour alone.
DOUBLE_RAYS: Final[dict[Color, list[list[int]]]] = {
    color: DIAG_RAYS[(color, 2)] for color in (WHITE, BLACK)
}
SINGLE_RAYS: Final[dict[Color, list[list[int]]]] = {
    color: DIAG_RAYS[(color, 1)] for color in (WHITE, BLACK)
}
# A stored search result. Every field is always present -- a position that has
# not been searched has no entry at all, rather than an entry full of Nones.
TTEntry = tuple[int, Move | None, str, int]

# Not Final, unlike the constants below: the tests and scripts/bench.py reassign
# these to run the search to a fixed depth, and a compiled build folds a Final
# into the code it generates, so marking them would make those patches silently
# do nothing. The same goes for MAX_SEARCH_DEPTH, TT_MAX_ENTRIES and
# EVAL_CACHE_MAX_ENTRIES.
MIN_SEARCH_DEPTH = 5
MILLISECONDS_PER_MOVE: float = 6000
MAX_MILLISECONDS_PER_MOVE: float = 10000

# Hard ceiling on iterative-deepening depth: a backstop so a cheaply-resolved position
# cannot spin the depth counter through the whole move budget. Not Final, for the
# same reason as the budgets above -- a test lowers it to check the cap holds.
MAX_SEARCH_DEPTH = 64
# A search value of at least this magnitude marks a proven forced win/loss (it is well
# above any reachable heuristic score). Deeper search cannot change such a result, so
# iterative deepening stops as soon as one is found.
MATE_THRESHOLD: Final = WIN_VALUE // 2
# The search works in integers throughout, because evaluate() does; this stands
# in for the infinite initial window. Kept clear of WIN_VALUE so a proven win is
# still strictly inside it. A float window would otherwise be the one thing
# forcing the search values to be floats -- which, compiled, means boxing a
# double into every transposition-table entry, and printing "56.0" where the
# interpreted engine prints "56".
INFINITY: Final = WIN_VALUE * 2

# How often the search calls back into on_tick (in nodes), so a front end can stay
# responsive without paying a callback per node.
TICK_INTERVAL_NODES: Final = 4096
# How often the move's time budget is checked (in nodes). time.time() plus the
# conversion to milliseconds costs more than most of a node's remaining bookkeeping,
# and checking it per node bought nothing: at the engine's node rate 512 nodes is a
# few milliseconds of overshoot against a 6000 ms budget.
TIME_CHECK_INTERVAL_NODES: Final = 512

# Bound the transposition table and eval cache so they cannot grow without limit
# across a long game; once full, the oldest entries are evicted first (FIFO).
TT_MAX_ENTRIES = 2_000_000
EVAL_CACHE_MAX_ENTRIES = 2_000_000


class ABTimeOut(Exception):
    pass


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
        # The TT is keyed on the board hash shifted left with the side to move in
        # bit 0: the hash alone collides for the same position with opposite sides to
        # move, whose minimax values differ. One int hashes far cheaper than the
        # (hash, colour-tuple) pair this used to be, and it is looked up twice a node.
        self.transposition_table: dict[int, TTEntry] = {}
        # Search-state attributes (also set in iterative_deepening); defaulted here so
        # alpha_beta can be called directly.
        self.search_start_time: int = 0
        self.min_search_depth_reached: bool = False
        self.completed_any_depth: bool = False
        # Leaf-evaluation cache. evaluate() is a pure, turn-independent function of the
        # board, so it can be memoized by state_hash across the whole game/search.
        self.eval_cache: dict[int, int] = {}
        # Killer moves (two per remaining-depth slot) and a table of cutoff counts,
        # both used only to order quiet moves.
        self.killers: list[list[Move | None]] = [
            [None, None] for _ in range(MAX_SEARCH_DEPTH + 2)
        ]
        self.history: dict[Move, int] = {}
        # Optional no-argument callback, invoked every TICK_INTERVAL_NODES nodes so a
        # front end can pump its event queue while the search runs. current_depth is
        # the iterative-deepening depth in progress, for the caller to display.
        self.on_tick: Callable[[], None] | None = None
        self.current_depth: int = 0
        self._tick_countdown: int = TICK_INTERVAL_NODES
        self._time_countdown: int = TIME_CHECK_INTERVAL_NODES
        # Per-move diagnostic counters, populated only when dev is on (see reset_stats).
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

    # Transposition table retrieval and storage

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
            # Prefer the entry searched to the greater depth; on a tie the newer
            # result replaces the old one.
            if existing[3] > depth:
                return
        elif len(table) >= TT_MAX_ENTRIES:
            del table[next(iter(table))]
        table[key] = (value, move, flag, depth)

    # Finding moves

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
        # Most nodes have no transposition-table move; an identity test then
        # replaces a tuple comparison per move.
        seeded = most_promising_move is not None
        for move in position.moves_list:
            _, target, tag = move
            if seeded and move == most_promising_move:
                check_first = [move]
            # Slides and transposes are tested first because they are the bulk
            # of a move list, and the killer slots can only hold one of those
            # two (alpha_beta records a cutoff move only for tags "S" and "T"),
            # so no other branch has to compare against them.
            elif tag == "S":
                if move == killer_1 or move == killer_2:
                    killer_moves.append(move)
                else:
                    # A slide is ranked by whether its destination ends up next
                    # to an enemy double (blocking it) and then an enemy single,
                    # counting up to two blocks. Only the first occupied square
                    # along each of the two diagonals matters.
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
        # Extended into one list rather than chained with `+`, which allocated a
        # fresh list per bucket and recopied everything ahead of it.
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

        # Terminate if you run out of time, but never before at least one full
        # depth has completed (so iterative_deepening always has a legal move).
        # Checked every TIME_CHECK_INTERVAL_NODES nodes, not every node.
        self._time_countdown -= 1
        if self._time_countdown <= 0:
            self._time_countdown = TIME_CHECK_INTERVAL_NODES
            move_time = milliseconds(time.time()) - self.search_start_time
            if self.completed_any_depth and (
                (move_time > MILLISECONDS_PER_MOVE and self.min_search_depth_reached)
                or move_time > MAX_MILLISECONDS_PER_MOVE
            ):
                raise ABTimeOut

        # Let a front end pump its event queue without blocking for the whole search.
        if self.on_tick is not None:
            self._tick_countdown -= 1
            if self._tick_countdown <= 0:
                self._tick_countdown = TICK_INTERVAL_NODES
                self.on_tick()

        if self.dev:
            self.nodes += 1
        old_alpha, old_beta = alpha, beta
        # Search for the position in the transposition table. If the search depth in
        # the TT is larger than the current search depth, then trust the TT entry.
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

        # Regular Alpha-Beta
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

        # White maximises, Black minimises. Branching on the side to move once keeps
        # two Python-level closure calls per move out of the hot loop.
        maximising = position.turn == WHITE
        value = -INFINITY if maximising else INFINITY
        best_move: Move | None = None
        # Check TT move first
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
                # Remember quiet moves that cut off, to try them earlier elsewhere.
                if tag == "S" or tag == "T":
                    slot = self.killers[depth]
                    if slot[0] != move:
                        slot[1] = slot[0]
                        slot[0] = move
                    self.history[move] = self.history.get(move, 0) + depth * depth
                break

        # Store the position in the TT
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
        # Seed with a safe fallback so a timeout before any depth completes still
        # returns a value (the depth-1 search is guaranteed to complete, however).
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
            # A proven forced win/loss cannot change with deeper search, and there is
            # no point searching past the depth ceiling, so stop rather than spend the
            # rest of the move budget re-deriving the same result.
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
        # If there is only one legal move, return it without searching. The flat
        # list makes this exact: one entry means one move, where the nested dict
        # needed a second check that one origin did not hide several targets.
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
