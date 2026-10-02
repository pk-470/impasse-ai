import time
from math import inf
from typing import Callable, Optional

from impasse.position import (
    COLOR_CODES,
    DIAG_INDICES,
    EMPTY,
    MOVE_DIRECTIONS,
    OPPOSITE_COLOR,
    SINGLE_CODE,
    WHITE,
    WIN_VALUE,
    Cell,
    Color,
    MoveTag,
    Position,
    _index,
)

Move = tuple[Cell, Optional[Cell], MoveTag]
TTEntry = tuple[float, Optional[Move], Optional[str], Optional[int]]

MIN_SEARCH_DEPTH = 5
MILLISECONDS_PER_MOVE = 6000
MAX_MILLISECONDS_PER_MOVE = 10000

# Hard ceiling on iterative-deepening depth: a backstop so a cheaply-resolved position
# cannot spin the depth counter through the whole move budget.
MAX_SEARCH_DEPTH = 64
# A search value of at least this magnitude marks a proven forced win/loss (it is well
# above any reachable heuristic score). Deeper search cannot change such a result, so
# iterative deepening stops as soon as one is found.
MATE_THRESHOLD = WIN_VALUE // 2

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
        # The TT is keyed on (state_hash, turn): the board hash alone collides for
        # the same position with opposite sides to move, whose minimax values differ.
        self.transposition_table: dict[tuple[int, Color], TTEntry] = {}
        # Search-state attributes (also set in iterative_deepening); defaulted here so
        # alpha_beta can be called directly.
        self.search_start_time: int = 0
        self.min_search_depth_reached: bool = False
        self.completed_any_depth: bool = False
        # Leaf-evaluation cache. evaluate() is a pure, turn-independent function of the
        # board, so it can be memoized by state_hash across the whole game/search.
        self.eval_cache: dict[int, int] = {}
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

    def tt_retrieve(self, position: Position) -> TTEntry:
        """
        Look up a position's transposition-table entry.

        Args:
            position: The position to look up.

        Returns:
            The stored (value, move, flag, depth) entry, or a (0.0, None, None,
            None) sentinel whose None depth marks a miss.
        """
        entry = self.transposition_table.get((position.state_hash, position.turn))
        if self.dev:
            self.tt_lookups += 1
            if entry is not None:
                self.tt_hits += 1
        if entry is None:
            return 0.0, None, None, None
        return entry

    def tt_store(
        self,
        position: Position,
        value: float,
        move: Optional[Move],
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
        key = (position.state_hash, position.turn)
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
        most_promising_move: Optional[Move] = None,
    ) -> list[Move]:
        """
        Order a position's legal moves to improve alpha-beta pruning.

        Args:
            position: The position whose moves to order.
            most_promising_move: A move to try first ahead of all others, if any.

        Returns:
            The legal moves as (origin, target, tag) tuples, most promising first:
            the most_promising_move, then forced crownings, bear offs, slides that
            block enemy doubles then singles (two blocks before one), potential
            crownings, transposes, and finally the remaining slides longest first.
        """
        check_first = []
        bear_offs = []
        crownings = []
        potential_crownings = []
        transposes = []
        slides_blocking_doubles_once = []
        slides_blocking_doubles_twice = []
        slides_blocking_singles_once = []
        slides_blocking_singles_twice = []
        other_slides = []
        all_legal_moves = position.all_legal_moves
        state = position.state
        enemy = OPPOSITE_COLOR[position.turn]
        enemy_double = COLOR_CODES[enemy][1]
        enemy_single = SINGLE_CODE[enemy]
        double_dirs = MOVE_DIRECTIONS[(position.turn, 2)]
        single_dirs = MOVE_DIRECTIONS[(position.turn, 1)]
        for origin, moves in all_legal_moves.items():
            for target, tag in moves.items():
                move = (origin, target, tag)
                if move == most_promising_move:
                    check_first = [move]
                elif tag == "C":
                    crownings.append(move)
                elif tag in ("B", "SB", "TB"):
                    bear_offs.append(move)
                elif tag in ("SC", "TC"):
                    potential_crownings.append(move)
                elif tag == "T":
                    transposes.append(move)
                elif tag == "S":
                    # A slide is ranked by whether its destination ends up next to an
                    # enemy double (blocking it) and then an enemy single, counting up
                    # to two blocks. The first occupied square along each diagonal is
                    # the only one that matters.
                    assert target is not None
                    target_index = _index(target)
                    blocks_double_once = False
                    blocks_double_twice = False
                    for direction in double_dirs:
                        for idx in DIAG_INDICES[(target_index, direction)]:
                            code = state[idx]
                            if code == EMPTY:
                                continue
                            if code == enemy_double:
                                if blocks_double_once:
                                    blocks_double_twice = True
                                else:
                                    blocks_double_once = True
                            break
                    if blocks_double_twice:
                        slides_blocking_doubles_twice.append(move)
                    elif blocks_double_once:
                        slides_blocking_doubles_once.append(move)
                    else:
                        blocks_single_once = False
                        blocks_single_twice = False
                        for direction in single_dirs:
                            for idx in DIAG_INDICES[(target_index, direction)]:
                                code = state[idx]
                                if code == EMPTY:
                                    continue
                                if code == enemy_single:
                                    if blocks_single_once:
                                        blocks_single_twice = True
                                    else:
                                        blocks_single_once = True
                                break
                        if blocks_single_twice:
                            slides_blocking_singles_twice.append(move)
                        elif blocks_single_once:
                            slides_blocking_singles_once.append(move)
                        else:
                            other_slides.append(move)

        other_slides.sort(reverse=True, key=lambda x: abs(x[0][0] - x[1][0]))
        return (
            check_first
            + crownings
            + bear_offs
            + slides_blocking_doubles_twice
            + slides_blocking_doubles_once
            + slides_blocking_singles_twice
            + slides_blocking_singles_once
            + potential_crownings
            + transposes
            + other_slides
        )

    def minimax_parameters(
        self, color: Color
    ) -> tuple[
        float,
        Callable[[float, float], bool],
        Callable[[float, float, float], tuple[float, float]],
    ]:
        """
        Build the starting search parameters for the player to move.

        Args:
            color: The side to move.

        Returns:
            A (start_value, is_better, update_window) triple: the worst-case
            initial value for the player, a predicate telling whether one value
            beats the current best for this player, and a function that tightens
            the (alpha, beta) window given a new value. White maximises, Black
            minimises.
        """
        if color == WHITE:
            return (
                -inf,
                lambda local_value, value: local_value > value,
                lambda alpha, beta, value: (max(alpha, value), beta),
            )
        else:
            return (
                inf,
                lambda local_value, value: local_value < value,
                lambda alpha, beta, value: (alpha, min(beta, value)),
            )

    def alpha_beta(
        self,
        position: Position,
        depth: int,
        alpha: float,
        beta: float,
    ) -> tuple[float, Optional[Move]]:
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
        move_time = milliseconds(time.time()) - self.search_start_time
        if self.completed_any_depth and (
            (move_time > MILLISECONDS_PER_MOVE and self.min_search_depth_reached)
            or move_time > MAX_MILLISECONDS_PER_MOVE
        ):
            raise ABTimeOut

        if self.dev:
            self.nodes += 1
        old_alpha, old_beta = alpha, beta
        # Search for the position in the transposition table. If the search depth in
        # the TT is larger than the current search depth, then trust the TT entry.
        tt_value, tt_move, tt_flag, tt_depth = self.tt_retrieve(position)
        if tt_depth is not None and tt_depth >= depth:
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

        start_value, value_test, alpha_beta_assignment = self.minimax_parameters(
            position.turn
        )
        value = start_value
        best_move: Optional[Move] = None
        # Check TT move first
        for origin, target, tag in self.ordered_moves(position, tt_move):
            new_position = position.new_position_after_move(origin, target, tag)
            if new_position.turn == position.turn:
                local_value, _ = self.alpha_beta(new_position, depth, alpha, beta)
            else:
                local_value, _ = self.alpha_beta(new_position, depth - 1, alpha, beta)
            if value_test(local_value, value):
                value = local_value
                best_move = (origin, target, tag)
            alpha, beta = alpha_beta_assignment(alpha, beta, value)
            if alpha >= beta:
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

    def iterative_deepening(
        self, position: Position
    ) -> tuple[int, float, Optional[Move]]:
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
        # Seed with a safe fallback so a timeout before any depth completes still
        # returns a value (the depth-1 search is guaranteed to complete, however).
        prev_search_depth, prev_value, prev_best_move = 0, position.evaluate(), None
        while True:
            if search_depth > MIN_SEARCH_DEPTH:
                self.min_search_depth_reached = True
            try:
                value, best_move = self.alpha_beta(
                    position,
                    search_depth,
                    -inf,
                    inf,
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
    ) -> tuple[Optional[Cell], Optional[Cell], Optional[MoveTag], bool]:
        """
        Choose the best move for the player to move and print its evaluation.

        Args:
            position: The position to move in.

        Returns:
            An (origin, target, tag, unique_move) tuple: the chosen move and a
            flag that is True when it was the only legal move (returned without
            searching). The first three are None if there is no legal move.
        """
        # If there is only one legal move, return it without searching.
        if len(position.all_legal_moves) == 1:
            origin, targets = next(iter(position.all_legal_moves.items()))
            if len(targets) == 1:
                target = next(iter(targets))
                tag = targets[target]
                value = position.evaluate()
                print(f"Alpha-Beta evaluation: {value} at depth 0 (one legal move)")
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
