from __future__ import annotations

import random
from mypy_extensions import i64, mypyc_attr
from typing import Literal, Optional, TypeAlias

Cell: TypeAlias = tuple[int, int]
Color: TypeAlias = tuple[int, int, int]
Piece: TypeAlias = tuple[Color, int]
# The board is a flat list of 64 int square codes (see PIECE_TO_CODE), indexed by
# row * 8 + col. A legacy dict[cell, (color, type) | None] is also accepted on input.
State: TypeAlias = list[int]
LegacyState: TypeAlias = dict[Cell, Optional[Piece]]
MoveTag: TypeAlias = Literal["S", "SB", "SC", "T", "TB", "TC", "C", "B"]
MoveDestination: TypeAlias = Optional[Cell]
MoveDict: TypeAlias = dict[MoveDestination, MoveTag]
LegalMoves: TypeAlias = dict[Cell, MoveDict]
# A single move as the search handles it: where from, where to (None for a bear
# off) and what kind. This is the primary form move generation produces.
Move: TypeAlias = tuple[Cell, MoveDestination, MoveTag]

WHITE: Color = (255, 255, 255)
BLACK: Color = (0, 0, 0)
OPPOSITE_COLOR: dict[Color, Color] = {WHITE: BLACK, BLACK: WHITE}

INITIAL_STATE: LegacyState = {
    (i, j): (WHITE, 1)
    if (i, j) in ((0, 0), (3, 1), (4, 0), (7, 1))
    else (WHITE, 2)
    if (i, j) in ((1, 7), (2, 6), (5, 7), (6, 6))
    else (BLACK, 1)
    if (i, j) in ((0, 6), (3, 7), (4, 6), (7, 7))
    else (BLACK, 2)
    if (i, j) in ((1, 1), (2, 0), (5, 1), (6, 0))
    else None
    for i in range(8)
    for j in range(8)
    if (i + j) % 2 == 0
}

# Each square holds a compact int code instead of a (color, type) tuple.
EMPTY: int = 0
PIECE_TO_CODE: dict[Optional[Piece], int] = {
    None: EMPTY,
    (WHITE, 1): 1,
    (WHITE, 2): 2,
    (BLACK, 1): 3,
    (BLACK, 2): 4,
}
CODE_TO_PIECE: dict[int, Optional[Piece]] = {c: p for p, c in PIECE_TO_CODE.items()}
CODE_TYPE: dict[int, int] = {1: 1, 2: 2, 3: 1, 4: 2}
SINGLE_CODE: dict[Color, int] = {WHITE: 1, BLACK: 3}
COLOR_CODES: dict[Color, tuple[int, int]] = {WHITE: (1, 2), BLACK: (3, 4)}


def _index(cell: Cell) -> int:
    """Return the flat board index (row * 8 + col) of a cell."""
    return cell[0] * 8 + cell[1]


# The 32 dark squares in board order, and their flat indices.
DARK_CELLS: list[Cell] = list(INITIAL_STATE)
DARK_INDICES: list[int] = [_index(cell) for cell in DARK_CELLS]

INITIAL_STATE_FLAT: list[int] = [EMPTY] * 64
for _cell, _piece in INITIAL_STATE.items():
    INITIAL_STATE_FLAT[_index(_cell)] = PIECE_TO_CODE[_piece]


def _to_flat(state: State | LegacyState) -> State:
    """
    Normalise a board to the flat int representation.

    Args:
        state: Either a flat list of square codes (returned unchanged, so the
            caller's list becomes owned by the position) or a legacy
            dict[cell, (color, type) | None] board (converted to a fresh list).

    Returns:
        A length-64 list of square codes indexed by row * 8 + col.
    """
    if isinstance(state, list):
        return state
    flat = [EMPTY] * 64
    for cell, piece in state.items():
        flat[_index(cell)] = PIECE_TO_CODE[piece]
    return flat


HOME_ROW: dict[Color, tuple[Cell, ...]] = {
    WHITE: ((0, 0), (2, 0), (4, 0), (6, 0)),
    BLACK: ((1, 7), (3, 7), (5, 7), (7, 7)),
}
# Frozenset variant for fast membership tests in the hot evaluation path. The tuple
# above is kept for get_crownings, whose .update() order depends on iteration order.
HOME_ROW_SET: dict[Color, frozenset] = {
    color: frozenset(cells) for color, cells in HOME_ROW.items()
}
HOME_INDICES: dict[Color, frozenset] = {
    color: frozenset(_index(cell) for cell in cells)
    for color, cells in HOME_ROW.items()
}


MOVE_DIRECTIONS: dict[Piece, tuple[Cell, ...]] = {
    (WHITE, 1): ((1, 1), (-1, 1)),
    (WHITE, 2): ((1, -1), (-1, -1)),
    (BLACK, 1): ((1, -1), (-1, -1)),
    (BLACK, 2): ((1, 1), (-1, 1)),
}
# Same directions keyed by square code, for lookups from a board square.
MOVE_DIRECTIONS_BY_CODE: dict[int, tuple[Cell, ...]] = {
    PIECE_TO_CODE[piece]: dirs for piece, dirs in MOVE_DIRECTIONS.items()
}

DIAGONALS: dict[tuple[Cell, Cell], list[Cell]] = {
    ((i, j), d): [
        (i + s * d[0], j + s * d[1])
        for s in range(1, 8)
        if 0 <= i + s * d[0] < 8 and 0 <= j + s * d[1] < 8
    ]
    for i, j in INITIAL_STATE
    for checker in MOVE_DIRECTIONS
    for d in MOVE_DIRECTIONS[checker]
}
# Same diagonals expressed as flat indices, keyed by (origin index, direction), for
# the hot evaluation pathfinders.
DIAG_INDICES: dict[tuple[int, Cell], list[int]] = {
    (_index(cell), d): [_index(c) for c in cells]
    for (cell, d), cells in DIAGONALS.items()
}
# The same flat diagonals indexed by [origin index * 2 + direction-choice] for each
# colour and checker type, so move generation and move ordering index a list by a
# small int instead of building a (cell, direction) tuple and hashing it. Indexing a
# list beats hashing a tuple in bytecode and compiles to a load, so unlike the
# pathfinders' native forms this needs no NATIVE branch.
DIAG_RAYS: dict[Piece, list[list[int]]] = {
    piece: [
        DIAG_INDICES.get((anchor, MOVE_DIRECTIONS[piece][i]), [])
        for anchor in range(64)
        for i in (0, 1)
    ]
    for piece in MOVE_DIRECTIONS
}
# Slide generation needs the destination cell (the move dicts are keyed by cell) as
# well as its index, so it reads (cell, index) pairs and never calls _index.
SLIDE_RAYS: dict[int, list[list[tuple[Cell, int]]]] = {
    PIECE_TO_CODE[piece]: [
        [
            (DARK_CELLS[DARK_INDICES.index(idx)], idx)
            for idx in DIAG_INDICES.get((anchor, MOVE_DIRECTIONS[piece][i]), [])
        ]
        for anchor in range(64)
        for i in (0, 1)
    ]
    for piece in MOVE_DIRECTIONS
}
# Home rows as 64-byte 0/1 tables, so a membership test is one indexed byte read
# instead of a frozenset hash. Fast in both builds, unlike HOME_MASK.
HOME_TABLE: dict[Color, bytes] = {
    color: bytes(1 if idx in indices else 0 for idx in range(64))
    for color, indices in HOME_INDICES.items()
}
# Side-to-move as a bit, for composing a single-int transposition-table key.
# Keyed by value, not identity: an unpickled turn colour is an equal but distinct
# tuple (gui.py's undo path assigns one), so `turn is WHITE` would be wrong there.
TURN_BIT: dict[Color, int] = {WHITE: 0, BLACK: 1}
# The same flat diagonals indexed directly by [anchor][direction-choice 0/1] for each
# colour, so the hot pathfinders avoid building a tuple key and hashing it on every
# recursive step. BEAR_DIAGS follows a double's two move directions, CROWN_DIAGS a
# single's. Diagonal moves keep a piece on dark squares, so only dark anchors are ever
# read; light anchors map to an empty list.
BEAR_DIAGS: dict[Color, list[list[list[int]]]] = {
    color: [
        [DIAG_INDICES.get((anchor, MOVE_DIRECTIONS[(color, 2)][i]), []) for i in (0, 1)]
        for anchor in range(64)
    ]
    for color in (WHITE, BLACK)
}
CROWN_DIAGS: dict[Color, list[list[list[int]]]] = {
    color: [
        [DIAG_INDICES.get((anchor, MOVE_DIRECTIONS[(color, 1)][i]), []) for i in (0, 1)]
        for anchor in range(64)
    ]
    for color in (WHITE, BLACK)
}

# True when this module was compiled by mypyc (its __file__ is then the extension,
# not the source). The pathfinders exist in two flavours and this picks between
# them: see _bear_off_walk_pure / _bear_off_walk_native below. Derived rather than
# configured, so a compiled build and its flag can never disagree.
NATIVE: bool = not __file__.endswith((".py", ".pyc"))

# Native-flavour lookups. A home row becomes a 64-bit occupancy bitmask (one
# shift+test rather than a frozenset hash), and the diagonals are flattened to one
# `bytes` ray per (anchor, direction) at index anchor * 2 + i, so a leg reads a
# native byte instead of indexing a list of lists of boxed ints. Both forms are
# built either way; the unused one costs a few KB and no time.
def _signed64(mask: int) -> int:
    """Reinterpret a 64-bit mask as a signed two's-complement int.

    Index 63 is a home square, so an unsigned mask overflows mypyc's native
    i64. Bit extraction by arithmetic shift is identical either way.
    """
    return mask - (1 << 64) if mask >= (1 << 63) else mask


HOME_MASK: dict[Color, int] = {
    color: _signed64(sum(1 << idx for idx in indices))
    for color, indices in HOME_INDICES.items()
}
BEAR_RAYS: dict[Color, list[bytes]] = {
    color: [bytes(diags[anchor][i]) for anchor in range(64) for i in (0, 1)]
    for color, diags in BEAR_DIAGS.items()
}
CROWN_RAYS: dict[Color, list[bytes]] = {
    color: [bytes(diags[anchor][i]) for anchor in range(64) for i in (0, 1)]
    for color, diags in CROWN_DIAGS.items()
}

# Sentinel "no path found" score, below any score a real path can reach.
_NO_PATH: int = -1_000

# Sentinel for the `winner` argument: read the winner off the board rather than
# taking the caller's word for it. Distinct from None, which means "nobody has won".
DERIVE_WINNER: object = object()

# Parameters and weights for evaluation
DOUBLES_PATHS_MAX: int = 10
SINGLES_PATHS_MAX: int = 10

CHECKERS_COUNT_WEIGHT: int = 120
DOUBLES_PATHS_WEIGHT: int = 8
SINGLES_PATHS_WEIGHT: int = 2
DOUBLES_WEIGHT: int = 1
# Score of a decided position. Kept far above any reachable heuristic score so that a
# real win/loss always outranks a heuristic line and can be detected as terminal.
WIN_VALUE: int = 100_000


# Zobrist hashing: a random 64-bit id for each (square index, code) combination,
# used to hash board states for the transposition table. Stored as a 2D list indexed
# by [square index][code] so the hot path indexes twice instead of hashing a tuple key.
random.seed(42)
rand_ids: list[list[int]] = [[0] * 5 for _ in range(64)]
for _idx in DARK_INDICES:
    for _code in range(5):
        rand_ids[_idx][_code] = random.getrandbits(64)


def _moves_to_dict(moves: list[Move]) -> LegalMoves:
    """
    Group a flat move list into the nested dict form.

    Args:
        moves: Moves as (origin, target, tag) triples.

    Returns:
        A dict mapping each origin cell to its {target: tag} moves, origins in
        first-seen order.
    """
    grouped: LegalMoves = {}
    for origin, target, tag in moves:
        entry = grouped.get(origin)
        if entry is None:
            grouped[origin] = {target: tag}
        else:
            entry[target] = tag
    return grouped


def _make_state_hash(state: list[int]) -> int:
    """
    Compute the Zobrist hash of a full board from scratch.

    Args:
        state: A flat list of square codes.

    Returns:
        The 64-bit Zobrist hash of the occupied configuration.
    """
    state_hash = 0
    for idx in DARK_INDICES:
        state_hash ^= rand_ids[idx][state[idx]]
    return state_hash


def _bear_off_walk_pure(
    state: list[int],
    single: int,
    home: frozenset,
    diags: list[list[list[int]]],
    anchor: int,
    i: int,
    best: int,
    steps: int,
    prev_empty: bool,
    changed_dir: bool,
) -> int:
    """
    Walk one double's bear-off paths and return the best score found.

    The pure-Python flavour, used when this module is interpreted: a frozenset
    home row and the nested-list diagonals are the fastest forms in bytecode.
    Must stay behaviour-identical to _bear_off_walk_native (see test_dual.py).

    Args:
        state: The flat board to walk.
        single: The square code of the moving player's single.
        home: The moving player's home-row indices.
        diags: The player's per-anchor double-move diagonals.
        anchor: The board index the current leg starts from.
        i: Which of the double's two move directions to follow.
        best: The best score found for this double so far (_NO_PATH if none).
        steps: Steps taken so far on the current path.
        prev_empty: Whether the previous square stepped through was empty.
        changed_dir: Whether the leg began by changing direction.

    Returns:
        The better of `best` and the best score reachable from this leg, a score
        being DOUBLES_PATHS_MAX minus the path's step count.
    """
    for idx in diags[anchor][i]:
        piece = state[idx]
        if piece == EMPTY:
            # A step is added if we move from a single
            # to an empty cell
            if not prev_empty:
                steps += 1
                prev_empty = True
            changed_dir = False
            # Add one step before changing direction
            new_steps = steps + 1
        elif piece == single:
            # A step is added if we encounter a single cell
            # unless it is after changing direction at an empty
            # cell (since then we have already added a step)
            if not (changed_dir and prev_empty):
                steps += 1
            changed_dir = False
            prev_empty = False
            # Change direction without adding step
            new_steps = steps
        else:
            break

        # Keep the shortest path to bear off found so far for this double
        # (converted into a score out of DOUBLES_PATHS_MAX)
        if idx in home and DOUBLES_PATHS_MAX - steps > best:
            best = DOUBLES_PATHS_MAX - steps
            break

        # Change direction. Steps only grow, so DOUBLES_PATHS_MAX - new_steps bounds
        # this leg: one that cannot beat `best` is dead and cutting it changes nothing.
        if DOUBLES_PATHS_MAX - new_steps > best:
            best = _bear_off_walk_pure(
                state,
                single,
                home,
                diags,
                idx,
                1 - i,
                best,
                new_steps,
                prev_empty,
                True,
            )

    return best


def _crown_walk_pure(
    state: list[int],
    home: frozenset,
    diags: list[list[list[int]]],
    anchor: int,
    i: int,
    best: int,
    steps: int,
) -> int:
    """
    Walk one single's paths to a crowning square and return the best score found.

    The pure-Python flavour; mirrors _crown_walk_native.

    Args:
        state: The flat board to walk.
        home: The opponent home-row indices (the crowning squares).
        diags: The player's per-anchor single-move diagonals.
        anchor: The board index the current leg starts from.
        i: Which of the single's two move directions to follow.
        best: The best score found for this single so far (_NO_PATH if none).
        steps: Steps taken so far on the current path.

    Returns:
        The better of `best` and the best score reachable from this leg, a score
        being SINGLES_PATHS_MAX minus the path's step count.
    """
    for idx in diags[anchor][i]:
        if state[idx] != EMPTY:
            break

        # Keep the shortest path to a crowning square found so far for this single
        # (converted into a score out of SINGLES_PATHS_MAX)
        if idx in home and SINGLES_PATHS_MAX - steps > best:
            best = SINGLES_PATHS_MAX - steps

        # Change direction (add 1 to steps), unless the bound rules the leg out.
        if SINGLES_PATHS_MAX - steps - 1 > best:
            best = _crown_walk_pure(state, home, diags, idx, 1 - i, best, steps + 1)

    return best

def _bear_off_walk_native(
    state: list[int],
    single: i64,
    home: i64,
    rays: list[bytes],
    anchor: i64,
    i: i64,
    best: i64,
    steps: i64,
    prev_empty: bool,
    changed_dir: bool,
) -> i64:
    """
    Walk one double's bear-off paths and return the best score found.

    The mypyc flavour, used when this module is compiled: native i64 scalars, a
    bitmask home row and flat `bytes` rays compile to C integer arithmetic with no
    boxing in the recursion, which is the engine's hottest loop. Interpreted this
    form is ~25% slower than _bear_off_walk_pure, which is why both exist.
    Must stay behaviour-identical to it (see test_dual.py).

    Args:
        state: The flat board to walk.
        single: The square code of the moving player's single.
        home: The moving player's home row as a 64-bit occupancy bitmask.
        rays: The player's flat double-move rays, indexed anchor * 2 + i.
        anchor: The board index the current leg starts from.
        i: Which of the double's two move directions to follow.
        best: The best score found for this double so far (_NO_PATH if none).
        steps: Steps taken so far on the current path.
        prev_empty: Whether the previous square stepped through was empty.
        changed_dir: Whether the leg began by changing direction.

    Returns:
        The better of `best` and the best score reachable from this leg, a score
        being DOUBLES_PATHS_MAX minus the path's step count.
    """
    ray = rays[anchor * 2 + i]
    k: i64 = 0
    length: i64 = len(ray)
    while k < length:
        idx: i64 = ray[k]
        k += 1
        piece: i64 = state[idx]
        new_steps: i64 = 0
        if piece == 0:
            # A step is added if we move from a single
            # to an empty cell
            if not prev_empty:
                steps += 1
                prev_empty = True
            changed_dir = False
            # Add one step before changing direction
            new_steps = steps + 1
        elif piece == single:
            # A step is added if we encounter a single cell
            # unless it is after changing direction at an empty
            # cell (since then we have already added a step)
            if not (changed_dir and prev_empty):
                steps += 1
            changed_dir = False
            prev_empty = False
            # Change direction without adding step
            new_steps = steps
        else:
            break

        # Keep the shortest path to bear off found so far for this double
        # (converted into a score out of DOUBLES_PATHS_MAX)
        if (home >> idx) & 1 and DOUBLES_PATHS_MAX - steps > best:
            best = DOUBLES_PATHS_MAX - steps
            break

        # Change direction. Steps only grow, so DOUBLES_PATHS_MAX - new_steps bounds
        # this leg: one that cannot beat `best` is dead and cutting it changes nothing.
        if DOUBLES_PATHS_MAX - new_steps > best:
            best = _bear_off_walk_native(
                state,
                single,
                home,
                rays,
                idx,
                1 - i,
                best,
                new_steps,
                prev_empty,
                True,
            )

    return best


def _crown_walk_native(
    state: list[int],
    home: i64,
    rays: list[bytes],
    anchor: i64,
    i: i64,
    best: i64,
    steps: i64,
) -> i64:
    """
    Walk one single's paths to a crowning square and return the best score found.

    The mypyc flavour; mirrors _crown_walk_pure.

    Args:
        state: The flat board to walk.
        home: The opponent home row (the crowning squares) as a bitmask.
        rays: The player's flat single-move rays, indexed anchor * 2 + i.
        anchor: The board index the current leg starts from.
        i: Which of the single's two move directions to follow.
        best: The best score found for this single so far (_NO_PATH if none).
        steps: Steps taken so far on the current path.

    Returns:
        The better of `best` and the best score reachable from this leg, a score
        being SINGLES_PATHS_MAX minus the path's step count.
    """
    ray = rays[anchor * 2 + i]
    k: i64 = 0
    length: i64 = len(ray)
    while k < length:
        idx: i64 = ray[k]
        k += 1
        if state[idx] != 0:
            break

        # Keep the shortest path to a crowning square found so far for this single
        # (converted into a score out of SINGLES_PATHS_MAX)
        if (home >> idx) & 1 and SINGLES_PATHS_MAX - steps > best:
            best = SINGLES_PATHS_MAX - steps

        # Change direction (add 1 to steps), unless the bound rules the leg out.
        if SINGLES_PATHS_MAX - steps - 1 > best:
            best = _crown_walk_native(state, home, rays, idx, 1 - i, best, steps + 1)

    return best


def _future_bear_offs_pure(
    state: list[int],
    single: int,
    home: frozenset,
    diags: list[list[list[int]]],
    doubles: list[int],
) -> int:
    """Sum the pure-flavour bear-off scores over a player's doubles."""
    total = 0
    for start in doubles:
        best = _bear_off_walk_pure(
            state, single, home, diags, start, 0, _NO_PATH, 0, False, False
        )
        best = _bear_off_walk_pure(
            state, single, home, diags, start, 1, best, 0, False, False
        )
        if best != _NO_PATH:
            total += best
    return total


def _future_bear_offs_native(
    state: list[int],
    single: i64,
    home: i64,
    rays: list[bytes],
    doubles: list[int],
) -> int:
    """Sum the native-flavour bear-off scores over a player's doubles."""
    total: i64 = 0
    for start in doubles:
        best = _bear_off_walk_native(
            state, single, home, rays, start, 0, _NO_PATH, 0, False, False
        )
        best = _bear_off_walk_native(
            state, single, home, rays, start, 1, best, 0, False, False
        )
        if best != _NO_PATH:
            total += best
    return int(total)


def _future_crowns_pure(
    state: list[int],
    home: frozenset,
    diags: list[list[list[int]]],
    singles: list[int],
) -> int:
    """Sum the pure-flavour crowning scores over a player's singles."""
    total = 0
    for start in singles:
        best = _crown_walk_pure(state, home, diags, start, 0, _NO_PATH, 1)
        best = _crown_walk_pure(state, home, diags, start, 1, best, 1)
        if best != _NO_PATH:
            total += best
    return total


def _future_crowns_native(
    state: list[int],
    home: i64,
    rays: list[bytes],
    singles: list[int],
) -> int:
    """Sum the native-flavour crowning scores over a player's singles."""
    total: i64 = 0
    for start in singles:
        best = _crown_walk_native(state, home, rays, start, 0, _NO_PATH, 1)
        best = _crown_walk_native(state, home, rays, start, 1, best, 1)
        if best != _NO_PATH:
            total += best
    return int(total)


@mypyc_attr(allow_interpreted_subclasses=True)
class Position:
    """
    A class to keep track of each position along with its available moves,
    and to calculate the effect of a move on a position.
    """

    # Declared here rather than in a single constructor path: there are two
    # (make_position and the clone fast path), and mypyc lays out the native
    # instance struct from these.
    state: State
    turn: Color
    checkers_total: dict[Color, int]
    winner: Optional[Color]
    state_hash: int
    # Legal moves are generated lazily on first access: most search nodes are
    # eval-only leaves that never need them. None means "not yet computed".
    # _moves_list is the primary form; _all_legal_moves is derived from it and
    # only built when something outside the search asks for it.
    _moves_list: Optional[list[Move]]
    _all_legal_moves: Optional[LegalMoves]

    def __init__(
        self,
        state: Optional[State | LegacyState] = None,
        turn: Optional[Color] = None,
        checkers_total: Optional[dict[Color, int]] = None,
        all_legal_moves: Optional[LegalMoves] = None,
        winner: Optional[Color] | object = DERIVE_WINNER,
        state_hash: Optional[int] = None,
        clone_of: Optional[Position] = None,
    ):
        # Fast path for copy(): every field is known, so skip make_position's
        # ladder of "derive this if it was not given" conditionals. One branch here
        # replaces about eight there, on a path the search takes once per node.
        if clone_of is not None:
            self.state = clone_of.state.copy()
            self.turn = clone_of.turn
            self._moves_list = None
            self._all_legal_moves = None
            self.checkers_total = clone_of.checkers_total.copy()
            self.winner = clone_of.winner
            self.state_hash = clone_of.state_hash
            return
        self.make_position(
            state, turn, checkers_total, all_legal_moves, winner, state_hash
        )

    def make_position(
        self,
        state: Optional[State | LegacyState] = None,
        turn: Optional[Color] = None,
        checkers_total: Optional[dict[Color, int]] = None,
        all_legal_moves: Optional[LegalMoves] = None,
        winner: Optional[Color] | object = DERIVE_WINNER,
        state_hash: Optional[int] = None,
    ) -> None:
        """
        Set up the position from the given data, deriving whatever is omitted.

        Args:
            state: The board, flat or legacy-dict; a fresh starting board if None.
            turn: The side to move; White if None.
            checkers_total: Each side's checker count; counted from the board if
                None.
            all_legal_moves: The position's legal moves; omit (None) to have the
                position work them out itself.
            winner: The winning side, or None for "nobody has won yet". Left at
                DERIVE_WINNER (the default) with an explicit state, the winner is
                read off the board instead: a side with no checkers left has won.
            state_hash: The board's Zobrist hash; computed from the board if None.
        """
        self.state = (
            INITIAL_STATE_FLAT.copy() if state is None else _to_flat(state)
        )
        self.turn = WHITE if turn is None else turn
        self._moves_list = None
        self._all_legal_moves = all_legal_moves
        self.checkers_total = (
            {WHITE: 12, BLACK: 12}
            if state is None
            else checkers_total
            if checkers_total
            else self.count_checkers()
        )
        self.winner = (
            None
            if state is None
            else winner  # type: ignore[assignment]
            if winner is not DERIVE_WINNER
            else WHITE
            if not self.checkers_total[WHITE]
            else BLACK
            if not self.checkers_total[BLACK]
            else None
        )
        self.state_hash = (
            state_hash if state_hash else _make_state_hash(self.state)
        )

    @property
    def moves_list(self) -> list[Move]:
        """Every legal move in the current position, as a flat list of triples."""
        if self._moves_list is None:
            self._moves_list = self.get_all_moves_list()
        return self._moves_list

    @property
    def all_legal_moves(self) -> LegalMoves:
        """Every legal move in the current position, keyed by origin cell."""
        if self._all_legal_moves is None:
            self._all_legal_moves = _moves_to_dict(self.moves_list)
        return self._all_legal_moves

    @all_legal_moves.setter
    def all_legal_moves(self, value: Optional[LegalMoves]) -> None:
        """Set the position's legal moves, or None to drop them."""
        self._all_legal_moves = value
        # Drop the flat form too; it is re-derived from the board on next access.
        self._moves_list = None

    def count_checkers(self) -> dict[Color, int]:
        """
        Count each player's checkers, a double counting as two.

        Returns:
            A dict mapping each colour to its total checker count.
        """
        white = black = 0
        state = self.state
        for idx in DARK_INDICES:
            code = state[idx]
            if code == 1:
                white += 1
            elif code == 2:
                white += 2
            elif code == 3:
                black += 1
            elif code == 4:
                black += 2
        return {WHITE: white, BLACK: black}

    def copy(self) -> Position:
        """
        Return an independent copy of this position.

        Returns:
            A new Position equal to this one in board, turn, checker counts,
            winner, legal moves and hash, sharing no mutable state with it.
        """
        # Legal moves are left unset on purpose: new_position_after_move re-derives
        # them, and most clones are eval-only leaves that never need them.
        return Position(clone_of=self)

    # Some shortcuts for various checks

    def is_occupied(self, cell: Cell) -> bool:
        """Return whether `cell` is on the board and holds a checker."""
        i, j = cell
        return 0 <= i < 8 and 0 <= j < 8 and self.state[_index(cell)] != EMPTY

    def is_single(self, cell: Cell) -> bool:
        """Return whether `cell` holds a single (uncrowned) checker."""
        i, j = cell
        return 0 <= i < 8 and 0 <= j < 8 and self.state[_index(cell)] in (1, 3)

    def is_occupied_of_color(self, cell: Cell, color: Color) -> bool:
        """Return whether `cell` holds a checker of the given colour."""
        i, j = cell
        return (
            0 <= i < 8 and 0 <= j < 8 and self.state[_index(cell)] in COLOR_CODES[color]
        )

    def is_occupied_single_of_color(self, cell: Cell, color: Color) -> bool:
        """Return whether `cell` holds a single checker of the given colour."""
        i, j = cell
        return (
            0 <= i < 8 and 0 <= j < 8 and self.state[_index(cell)] == SINGLE_CODE[color]
        )

    def piece_at(self, cell: Cell) -> Optional[Piece]:
        """
        Return the piece occupying a cell.

        Args:
            cell: The board cell to read.

        Returns:
            The piece as a (color, type) pair, or None if the cell is empty.
        """
        return CODE_TO_PIECE[self.state[_index(cell)]]

    # Impasse game functions

    def _append_slides(self, origin: Cell, out: list[Move]) -> None:
        """
        Append every slide available to the checker on a cell.

        Appends rather than returning a dict: generation runs on every internal
        search node and the per-origin dicts were allocated only to be flattened.

        Args:
            origin: The cell of the moving checker.
            out: The move list to append to. Tags are "S" for a regular slide,
                "SB" for a slide onto the player's home row (a bear off), or "SC"
                for a slide of a single onto the far row (enabling a crowning).
        """
        state = self.state
        origin_index = origin[0] * 8 + origin[1]
        origin_code = state[origin_index]
        rays = SLIDE_RAYS.get(origin_code)
        if rays is None:
            return
        origin_is_single = origin_code == 1 or origin_code == 3
        turn_home = HOME_TABLE[self.turn]
        far_home = HOME_TABLE[OPPOSITE_COLOR[self.turn]]
        base = origin_index * 2
        for i in (0, 1):
            for candidate, idx in rays[base + i]:
                if state[idx] != EMPTY:
                    break
                # Slide leading to bear off
                if turn_home[idx]:
                    out.append((origin, candidate, "SB"))
                # Slide leading to available crowning
                elif far_home[idx] and origin_is_single:
                    out.append((origin, candidate, "SC"))
                # Regular slide
                else:
                    out.append((origin, candidate, "S"))

    def _append_transposes(self, origin: Cell, out: list[Move]) -> None:
        """
        Append every transpose available to the crown on a cell.

        Args:
            origin: The cell of the moving crown.
            out: The move list to append to. Tags are "T" for a regular transpose,
                "TB" for one onto the player's home row (a bear off), or "TC" for
                one leaving a single on the far row (enabling a crowning).
        """
        state = self.state
        origin_index = origin[0] * 8 + origin[1]
        origin_code = state[origin_index]
        turn_home = HOME_TABLE[self.turn]
        far_home = HOME_TABLE[OPPOSITE_COLOR[self.turn]]
        origin_on_far_home = far_home[origin_index] == 1
        single = SINGLE_CODE[self.turn]
        for direction in MOVE_DIRECTIONS_BY_CODE.get(origin_code, ()):
            ci, cj = origin[0] + direction[0], origin[1] + direction[1]
            if not (0 <= ci < 8 and 0 <= cj < 8):
                continue
            candidate_index = ci * 8 + cj
            if state[candidate_index] == single:
                candidate = (ci, cj)
                # Transpose leading to bear off
                if turn_home[candidate_index]:
                    out.append((origin, candidate, "TB"))
                # Transpose leading to available crowning
                elif origin_on_far_home:
                    out.append((origin, candidate, "TC"))
                # Regular transpose
                else:
                    out.append((origin, candidate, "T"))

    def get_singles_except(self, target: Cell) -> list[Cell]:
        """
        List the mover's single checkers other than the one on a cell.

        Args:
            target: A cell to exclude from the result.

        Returns:
            The cells, other than target, holding a single checker of the player to
            move.
        """
        single = SINGLE_CODE[self.turn]
        state = self.state
        return [
            cell
            for cell in DARK_CELLS
            if cell != target and state[_index(cell)] == single
        ]

    def get_crownings(self) -> LegalMoves:
        """
        Find every mandatory crowning available to the player to move.

        Returns:
            A dict mapping each usable single checker's cell to {target: "C"}, where
            target is a single of the mover on the far home row that must be crowned.
            Empty if no crowning is available.
        """
        return _moves_to_dict(self.crownings_list())

    def crownings_list(self) -> list[Move]:
        """
        Find every mandatory crowning available to the player to move.

        Emitted grouped by origin, in board order. That matches the order the old
        nested-dict form flattened to in every case but one: when two of the
        mover's singles sit on the far row and crown each other, the dict put the
        first square's entry last, because its own pass excluded it and only a
        later target's pass created it. The two orders list the same moves either
        way, so this only changes which of two tied crownings the search returns.

        Returns:
            Each (helper single's cell, far-row single to crown, "C"). Empty if no
            crowning is available.
        """
        single = SINGLE_CODE[self.turn]
        state = self.state
        far_row = HOME_ROW[OPPOSITE_COLOR[self.turn]]
        # Cheap rejection first: no mover's single on the far row means no crowning
        # is owed. This runs after most move applications, so it must not pay for
        # the full board scan below (four reads instead of thirty-two).
        owed = False
        for target in far_row:
            if state[target[0] * 8 + target[1]] == single:
                owed = True
                break
        if not owed:
            return []

        out: list[Move] = []
        # One helper can crown more than one far-row single, so every origin keeps
        # all of its targets.
        for cell in DARK_CELLS:
            if state[cell[0] * 8 + cell[1]] != single:
                continue
            for target in far_row:
                if target != cell and state[target[0] * 8 + target[1]] == single:
                    out.append((cell, target, "C"))

        return out

    def get_other_moves(self) -> LegalMoves:
        """
        Find every legal move other than crownings for the player to move.

        Returns:
            A dict mapping each origin cell to its moves (destination cell to tag;
            the destination is None for a bear off). If the player has no slide or
            transpose, returns the forced bear-off ("B") of each of their checkers
            (the Impasse rule).
        """
        return _moves_to_dict(self.other_moves_list())

    def other_moves_list(self) -> list[Move]:
        """
        Find every legal move other than crownings for the player to move.

        Returns:
            The moves as triples (the target is None for a bear off). If the player
            has no slide or transpose, returns the forced bear-off ("B") of each of
            their checkers (the Impasse rule).
        """
        out: list[Move] = []
        turn = self.turn
        own = COLOR_CODES[turn]
        single = SINGLE_CODE[turn]
        state = self.state
        for cell in DARK_CELLS:
            code = state[cell[0] * 8 + cell[1]]
            if code in own:
                if code == single:
                    # Singles slide.
                    self._append_slides(cell, out)
                else:
                    # Crowns transpose as well as slide; transposes come first, as
                    # they did when this merged two dicts.
                    self._append_transposes(cell, out)
                    self._append_slides(cell, out)
        # Impasse
        if not out:
            return [
                (cell, None, "B")
                for cell in DARK_CELLS
                if state[cell[0] * 8 + cell[1]] in own
            ]
        return out

    def get_all_legal_moves(self) -> LegalMoves:
        """
        Find every legal move for the player to move.

        Returns:
            A dict mapping each origin cell to its moves (destination cell to
            tag). If the player owes a mandatory crowning, only the crownings
            are returned.
        """
        return _moves_to_dict(self.get_all_moves_list())

    def get_all_moves_list(self) -> list[Move]:
        """
        Find every legal move for the player to move, as a flat list.

        Returns:
            The moves as triples. If the player owes a mandatory crowning, only the
            crownings are returned.
        """
        crownings = self.crownings_list()
        if crownings:
            return crownings
        return self.other_moves_list()

    def apply_move(
        self, origin: Cell, target: MoveDestination, tag: MoveTag
    ) -> dict[Cell, int]:
        """
        Compute the squares a move changes, without mutating the position.

        Args:
            origin: The moving checker's cell.
            target: The destination cell, or None for a bear off.
            tag: The move tag.

        Returns:
            A dict mapping each changed cell to its new square code.
        """
        origin_index, origin_code, target_index, target_code = self._move_squares(
            origin, target, tag
        )
        if target_index < 0:
            return {origin: origin_code}
        assert target is not None
        return {origin: origin_code, target: target_code}

    def _move_squares(
        self, origin: Cell, target: MoveDestination, tag: MoveTag
    ) -> tuple[int, int, int, int]:
        """
        Compute the squares a move changes, as flat indices and codes.

        The same decision table as apply_move, which builds its dict from this, but
        without allocating one: the search applies millions of moves and only ever
        needs the indices. A bear off changes one square, every other move two.

        Args:
            origin: The moving checker's cell.
            target: The destination cell, or None for a bear off.
            tag: The move tag.

        Returns:
            (origin_index, origin_code, target_index, target_code), where
            target_index is -1 when the move changes only the origin square.
        """
        turn = self.turn
        single = SINGLE_CODE[turn]
        double = single + 1
        origin_index = origin[0] * 8 + origin[1]
        origin_type = CODE_TYPE[self.state[origin_index]]
        target_index = -1 if target is None else target[0] * 8 + target[1]
        # Slide
        if tag == "S" or tag == "SC":
            return origin_index, EMPTY, target_index, single + origin_type - 1
        # Slide + Bear off
        if tag == "SB":
            return origin_index, EMPTY, target_index, single
        # Transpose
        if tag == "T" or tag == "TC":
            return origin_index, single, target_index, double
        # Transpose + Bear off
        if tag == "TB":
            return origin_index, single, target_index, single
        # Crowning
        if tag == "C":
            return origin_index, EMPTY, target_index, double
        # Bear off: a single leaves the board, a double becomes a single.
        if tag == "B":
            return origin_index, EMPTY if origin_type == 1 else single, -1, 0
        raise ValueError(f"unknown move tag: {tag!r}")

    def change_turn(self) -> None:
        """
        Pass the turn to the other player.

        The new player's legal moves become available on the position, with any
        mandatory crownings offered ahead of other moves.
        """
        self.turn = OPPOSITE_COLOR[self.turn]
        self._moves_list = None
        self._all_legal_moves = None

    def check_for_crownings_and_change_turn(self) -> None:
        """
        Keep the current player on move if they owe a mandatory crowning,
        offering only those crownings; otherwise pass the turn to the other player.
        """
        crownings = self.crownings_list()
        if crownings:
            self._moves_list = crownings
            self._all_legal_moves = None
        else:
            self.change_turn()

    def update(self, state_update: dict[Cell, int], tag: MoveTag) -> None:
        """
        Apply a move's square changes and advance the game state.

        Args:
            state_update: The changed cells mapped to their new square codes, as
                returned by apply_move.
            tag: The tag of the move being applied.
        """
        state = self.state
        for cell, new_code in state_update.items():
            idx = _index(cell)
            row = rand_ids[idx]
            self.state_hash ^= row[state[idx]]
            self.state_hash ^= row[new_code]
            state[idx] = new_code
        self._advance(tag)

    def _write_square(self, idx: int, new_code: int) -> None:
        """Set one square and fold the change into the Zobrist hash."""
        row = rand_ids[idx]
        self.state_hash ^= row[self.state[idx]]
        self.state_hash ^= row[new_code]
        self.state[idx] = new_code

    def _advance(self, tag: MoveTag) -> None:
        """
        Advance turn, winner and pending crownings after a move's squares are set.

        Split out of update() so the search can write squares from the packed form
        of a move and still share this one copy of the turn-flow rules.

        Args:
            tag: The tag of the move just applied.
        """
        # Bear off
        if tag == "B":
            self.checkers_total[self.turn] -= 1
            # Check for win
            if not self.checkers_total[self.turn]:
                self.winner = self.turn
                # A decided position has no moves to offer.
                self._moves_list = []
                self._all_legal_moves = None
            # Might make crowning possible
            else:
                self.check_for_crownings_and_change_turn()
        # Slide + Bear off
        elif tag == "SB":
            self.checkers_total[self.turn] -= 1
            # Might make crowning possible
            self.check_for_crownings_and_change_turn()
        # Transpose + Bear off. Leaves a new single on the origin square, which is the
        # rules' "come to have another on-board single" crowning trigger.
        elif tag == "TB":
            self.checkers_total[self.turn] -= 1
            self.check_for_crownings_and_change_turn()
        # Potential crowning
        elif tag in ("SC", "TC"):
            self.check_for_crownings_and_change_turn()
        # A crowning can leave another mandatory crowning for the same player,
        # so re-check before changing turn
        elif tag == "C":
            self.check_for_crownings_and_change_turn()
        # Change turn after a regular slide or regular transpose
        elif tag in ("S", "T"):
            self.change_turn()

    def new_position_after_move(
        self, origin: Cell, target: MoveDestination, tag: MoveTag
    ) -> Position:
        """
        Apply a move to a copy of this position and return it.

        Args:
            origin: The moving checker's cell.
            target: The destination cell, or None for a bear off.
            tag: The move tag.

        Returns:
            A new Position with the move applied and the game advanced; this
            position is left unchanged.
        """
        new_position = Position(clone_of=self)
        oi, oc, ti, tc = new_position._move_squares(origin, target, tag)
        new_position._write_square(oi, oc)
        if ti >= 0:
            new_position._write_square(ti, tc)
        new_position._advance(tag)
        return new_position

    def future_bear_offs(self, color: Color, doubles: list[int]) -> int:
        """
        Score a player's prospects of bearing off their double checkers.

        Args:
            color: The player to score.
            doubles: The board indices of that player's double checkers.

        Returns:
            The combined shortest-path-to-bear-off score over those doubles.
        """
        if NATIVE:
            return _future_bear_offs_native(
                self.state, SINGLE_CODE[color], HOME_MASK[color],
                BEAR_RAYS[color], doubles,
            )
        return _future_bear_offs_pure(
            self.state, SINGLE_CODE[color], HOME_INDICES[color],
            BEAR_DIAGS[color], doubles,
        )

    def future_crowns(self, color: Color, singles: list[int]) -> int:
        """
        Score a player's prospects of crowning their single checkers.

        Args:
            color: The player to score.
            singles: The board indices of that player's single checkers.

        Returns:
            The combined shortest-path-to-crowning score over those singles.
        """
        enemy = OPPOSITE_COLOR[color]
        if NATIVE:
            return _future_crowns_native(
                self.state, HOME_MASK[enemy], CROWN_RAYS[color], singles
            )
        return _future_crowns_pure(
            self.state, HOME_INDICES[enemy], CROWN_DIAGS[color], singles
        )

    def evaluate(self) -> int:
        """
        Score the position from White's perspective.

        A decided position scores +WIN_VALUE if White has won and -WIN_VALUE if Black
        has (a magnitude kept well above any heuristic score, so a real win always
        outranks a heuristic line). Otherwise the score combines: each player's checker
        count (falling back to their double count when the counts are equal), the number
        and length of each player's paths towards bearing off, and towards crowning.

        Returns:
            The integer evaluation; positive favours White, negative favours Black.
        """
        if self.winner is not None:
            return WIN_VALUE if self.winner == WHITE else -WIN_VALUE
        # Single pass over the board, bucketing piece indices by colour and type,
        # instead of four full-board scans (one per future_* call).
        white_singles: list[int] = []
        white_doubles: list[int] = []
        black_singles: list[int] = []
        black_doubles: list[int] = []
        state = self.state
        for idx in DARK_INDICES:
            code = state[idx]
            if code == 1:
                white_singles.append(idx)
            elif code == 2:
                white_doubles.append(idx)
            elif code == 3:
                black_singles.append(idx)
            elif code == 4:
                black_doubles.append(idx)

        dwpw = self.future_bear_offs(WHITE, white_doubles)
        dwpb = self.future_bear_offs(BLACK, black_doubles)
        checkers_count = self.checkers_total[BLACK] - self.checkers_total[WHITE]
        if checkers_count:
            value = CHECKERS_COUNT_WEIGHT * checkers_count
        else:
            value = DOUBLES_WEIGHT * (len(white_doubles) - len(black_doubles))
        doubles_path_score = dwpw - dwpb
        singles_path_score = self.future_crowns(
            WHITE, white_singles
        ) - self.future_crowns(BLACK, black_singles)
        value += (
            DOUBLES_PATHS_WEIGHT * doubles_path_score
            + SINGLES_PATHS_WEIGHT * singles_path_score
        )
        return value