from __future__ import annotations

import random
from typing import Final, Literal

from mypy_extensions import i64, mypyc_attr

type Cell = tuple[int, int]
type Color = tuple[int, int, int]
type Piece = tuple[Color, int]
# The board is a flat list of 64 int square codes (see PIECE_TO_CODE), indexed by
# row * 8 + col. A legacy dict[cell, (color, type) | None] is also accepted on input.
type State = list[int]
type LegacyState = dict[Cell, Piece | None]
type MoveTag = Literal["S", "SB", "SC", "T", "TB", "TC", "C", "B"]
type MoveDestination = Cell | None
type MoveDict = dict[MoveDestination, MoveTag]
type LegalMoves = dict[Cell, MoveDict]
# A single move as the search handles it: where from, where to (None for a bear
# off) and what kind. This is the primary form move generation produces.
type Move = tuple[Cell, MoveDestination, MoveTag]

WHITE: Final[Color] = (255, 255, 255)
BLACK: Final[Color] = (0, 0, 0)
OPPOSITE_COLOR: Final[dict[Color, Color]] = {WHITE: BLACK, BLACK: WHITE}

INITIAL_STATE: Final[LegacyState] = {
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
EMPTY: Final = 0
PIECE_TO_CODE: Final[dict[Piece | None, int]] = {
    None: EMPTY,
    (WHITE, 1): 1,
    (WHITE, 2): 2,
    (BLACK, 1): 3,
    (BLACK, 2): 4,
}
CODE_TO_PIECE: Final[dict[int, Piece | None]] = {c: p for p, c in PIECE_TO_CODE.items()}
# Checker type (1 single, 2 crown) per square code, as a table indexed by the
# code rather than a dict keyed by it. Index 0 (empty) has no type.
CODE_TYPE: Final[bytes] = bytes((0, 1, 2, 1, 2))
SINGLE_CODE: Final[dict[Color, int]] = {WHITE: 1, BLACK: 3}
COLOR_CODES: Final[dict[Color, tuple[int, int]]] = {WHITE: (1, 2), BLACK: (3, 4)}


def _index(cell: Cell) -> int:
    """Return the flat board index (row * 8 + col) of a cell."""
    return cell[0] * 8 + cell[1]


# The 32 dark squares in board order, and their flat indices.
DARK_CELLS: Final[list[Cell]] = list(INITIAL_STATE)
DARK_INDICES: Final[list[int]] = [_index(cell) for cell in DARK_CELLS]
# The inverse of _index, so a scan over DARK_INDICES can name the cell it is on
# without the arithmetic. Cells are compared and hashed by value everywhere (and
# never with `is` -- they cross pickle boundaries, and compiled, mypyc re-boxes a
# tuple on its way into a list, so even DARK_CELLS and SLIDE_RAYS do not share
# objects there), so only the values here matter.
CELL_AT_INDEX: Final[list[Cell]] = [(idx // 8, idx % 8) for idx in range(64)]
for _cell in DARK_CELLS:
    CELL_AT_INDEX[_index(_cell)] = _cell

INITIAL_STATE_FLAT: Final[list[int]] = [EMPTY] * 64
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


HOME_ROW: Final[dict[Color, tuple[Cell, ...]]] = {
    WHITE: ((0, 0), (2, 0), (4, 0), (6, 0)),
    BLACK: ((1, 7), (3, 7), (5, 7), (7, 7)),
}
HOME_INDICES: Final[dict[Color, frozenset]] = {
    color: frozenset(_index(cell) for cell in cells)
    for color, cells in HOME_ROW.items()
}


MOVE_DIRECTIONS: Final[dict[Piece, tuple[Cell, ...]]] = {
    (WHITE, 1): ((1, 1), (-1, 1)),
    (WHITE, 2): ((1, -1), (-1, -1)),
    (BLACK, 1): ((1, -1), (-1, -1)),
    (BLACK, 2): ((1, 1), (-1, 1)),
}
DIAGONALS: Final[dict[tuple[Cell, Cell], list[Cell]]] = {
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
DIAG_INDICES: Final[dict[tuple[int, Cell], list[int]]] = {
    (_index(cell), d): [_index(c) for c in cells]
    for (cell, d), cells in DIAGONALS.items()
}
# The same flat diagonals indexed by [origin index * 2 + direction-choice] for each
# colour and checker type, so move generation and move ordering index a list by a
# small int instead of building a (cell, direction) tuple and hashing it. Indexing a
# list beats hashing a tuple in bytecode and compiles to a load, so unlike the
# pathfinders' native forms this needs no NATIVE branch.
DIAG_RAYS: Final[dict[Piece, list[list[int]]]] = {
    piece: [
        DIAG_INDICES.get((anchor, MOVE_DIRECTIONS[piece][i]), [])
        for anchor in range(64)
        for i in (0, 1)
    ]
    for piece in MOVE_DIRECTIONS
}
# Slide generation needs the destination cell (the move dicts are keyed by cell) as
# well as its index, so it reads (cell, index) pairs and never calls _index.
SLIDE_RAYS: Final[dict[int, list[list[tuple[Cell, int]]]]] = {
    PIECE_TO_CODE[piece]: [
        [
            (CELL_AT_INDEX[idx], idx)
            for idx in DIAG_INDICES.get((anchor, MOVE_DIRECTIONS[piece][i]), [])
        ]
        for anchor in range(64)
        for i in (0, 1)
    ]
    for piece in MOVE_DIRECTIONS
}
# Home rows as 64-byte 0/1 tables, so a membership test is one indexed byte read
# instead of a frozenset hash. Fast in both builds, unlike HOME_MASK.
HOME_TABLE: Final[dict[Color, bytes]] = {
    color: bytes(1 if idx in indices else 0 for idx in range(64))
    for color, indices in HOME_INDICES.items()
}
# Side-to-move as a bit, for composing a single-int transposition-table key.
# Keyed by value, not identity: an unpickled turn colour is an equal but distinct
# tuple (gui.py's undo path assigns one), so `turn is WHITE` would be wrong there.
TURN_BIT: Final[dict[Color, int]] = {WHITE: 0, BLACK: 1}
# The same flat diagonals indexed directly by [anchor][direction-choice 0/1] for each
# colour, so the hot pathfinders avoid building a tuple key and hashing it on every
# recursive step. BEAR_DIAGS follows a double's two move directions, CROWN_DIAGS a
# single's. Diagonal moves keep a piece on dark squares, so only dark anchors are ever
# read; light anchors map to an empty list.
BEAR_DIAGS: Final[dict[Color, list[list[list[int]]]]] = {
    color: [
        [DIAG_INDICES.get((anchor, MOVE_DIRECTIONS[(color, 2)][i]), []) for i in (0, 1)]
        for anchor in range(64)
    ]
    for color in (WHITE, BLACK)
}
CROWN_DIAGS: Final[dict[Color, list[list[list[int]]]]] = {
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
NATIVE: Final[bool] = not __file__.endswith((".py", ".pyc"))

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


HOME_MASK: Final[dict[Color, int]] = {
    color: _signed64(sum(1 << idx for idx in indices))
    for color, indices in HOME_INDICES.items()
}
# Every (anchor, direction) slot, numbered anchor * 2 + direction, gets a fixed
# RAY_STRIDE-byte span of this table holding its squares in order and then
# RAY_END. A diagonal is at most 7 squares long, so a span always has room for
# the sentinel. One flat `bytes` rather than a list of 128 of them: a walker
# step is then a single byte read, with no list index, no len() call, and no
# per-leg object for the caller to pass and the callee to release.
RAY_STRIDE: Final = 8
RAY_END: Final = 255


def _flat_rays(diags: list[list[list[int]]]) -> bytes:
    """
    Pack per-anchor diagonals into fixed-stride, sentinel-terminated spans.

    Args:
        diags: Diagonals indexed [anchor][direction] as flat square indices.

    Returns:
        A bytes table to be read as table[slot * RAY_STRIDE + step].
    """
    data = bytearray([RAY_END] * (128 * RAY_STRIDE))
    for anchor in range(64):
        for i in (0, 1):
            ray = diags[anchor][i]
            assert len(ray) < RAY_STRIDE, "a diagonal must leave room for RAY_END"
            base = (anchor * 2 + i) * RAY_STRIDE
            data[base : base + len(ray)] = bytes(ray)
    return bytes(data)


BEAR_RAYS: Final[dict[Color, bytes]] = {
    color: _flat_rays(diags) for color, diags in BEAR_DIAGS.items()
}
CROWN_RAYS: Final[dict[Color, bytes]] = {
    color: _flat_rays(diags) for color, diags in CROWN_DIAGS.items()
}

# The same tables split per colour. The fused evaluation loop below walks every
# piece of both colours, so it reads these globals instead of hashing a Color on
# each one; the dicts above stay for callers that have a colour in hand.
BEAR_DIAGS_WHITE: Final = BEAR_DIAGS[WHITE]
BEAR_DIAGS_BLACK: Final = BEAR_DIAGS[BLACK]
CROWN_DIAGS_WHITE: Final = CROWN_DIAGS[WHITE]
CROWN_DIAGS_BLACK: Final = CROWN_DIAGS[BLACK]
HOME_INDICES_WHITE: Final = HOME_INDICES[WHITE]
HOME_INDICES_BLACK: Final = HOME_INDICES[BLACK]
BEAR_RAYS_WHITE: Final = BEAR_RAYS[WHITE]
BEAR_RAYS_BLACK: Final = BEAR_RAYS[BLACK]
CROWN_RAYS_WHITE: Final = CROWN_RAYS[WHITE]
CROWN_RAYS_BLACK: Final = CROWN_RAYS[BLACK]
HOME_MASK_WHITE: Final = HOME_MASK[WHITE]
HOME_MASK_BLACK: Final = HOME_MASK[BLACK]
# The dark squares as a `bytes` table, so the native evaluation loop reads each
# index as a native int rather than unboxing a list element.
DARK_BYTES: Final = bytes(DARK_INDICES)

# Sentinel "no path found" score, below any score a real path can reach.
_NO_PATH: Final = -1_000

# Sentinel for the `winner` argument: read the winner off the board rather than
# taking the caller's word for it. Distinct from None, which means "nobody has won".
DERIVE_WINNER: Final[object] = object()

# Parameters and weights for evaluation
DOUBLES_PATHS_MAX: Final = 10
SINGLES_PATHS_MAX: Final = 10

CHECKERS_COUNT_WEIGHT: Final = 120
DOUBLES_PATHS_WEIGHT: Final = 8
SINGLES_PATHS_WEIGHT: Final = 2
DOUBLES_WEIGHT: Final = 1
# Score of a decided position. Kept far above any reachable heuristic score so that a
# real win/loss always outranks a heuristic line and can be detected as terminal.
WIN_VALUE: Final = 100_000


# Zobrist hashing: a random id for each (square index, code) combination, used to
# hash board states for the transposition table. Stored as a 2D list indexed by
# [square index][code] so the hot path indexes twice instead of hashing a tuple key.
#
# 61 bits, not 64: compiled, an int that fits in a tagged-pointer immediate is
# machine arithmetic, while a wider one is a heap PyLong, and every applied move
# XORs the hash four times and then derives the transposition-table key from it
# (state_hash * 2 + turn bit, which stays inside the immediate range too). The
# birthday collision chance over a full transposition table is still ~1e-6.
ZOBRIST_BITS: Final = 61
random.seed(42)
rand_ids: Final[list[list[int]]] = [[0] * 5 for _ in range(64)]
for _idx in DARK_INDICES:
    for _code in range(5):
        rand_ids[_idx][_code] = random.getrandbits(ZOBRIST_BITS)


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
    board: bytes,
    single: i64,
    home: i64,
    rays: bytes,
    slot: i64,
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
        board: The flat board to walk, as `bytes`. mypyc compiles a `bytes`
            index to a native byte read where `list[int]` has to unbox an
            object, and this is the engine's hottest read.
        single: The square code of the moving player's single.
        home: The moving player's home row as a 64-bit occupancy bitmask.
        rays: The player's flat double-move ray table (see _flat_rays).
        slot: Which ray to follow, as anchor * 2 + direction.
        best: The best score found for this double so far (_NO_PATH if none).
        steps: Steps taken so far on the current path.
        prev_empty: Whether the previous square stepped through was empty.
        changed_dir: Whether the leg began by changing direction.

    Returns:
        The better of `best` and the best score reachable from this leg, a score
        being DOUBLES_PATHS_MAX minus the path's step count.
    """
    k: i64 = slot * RAY_STRIDE
    while True:
        idx: i64 = rays[k]
        if idx == RAY_END:
            break
        k += 1
        piece: i64 = board[idx]
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
                board,
                single,
                home,
                rays,
                idx * 2 + (1 - (slot & 1)),
                best,
                new_steps,
                prev_empty,
                True,
            )

    return best


def _crown_walk_native(
    board: bytes,
    home: i64,
    rays: bytes,
    slot: i64,
    best: i64,
    steps: i64,
) -> i64:
    """
    Walk one single's paths to a crowning square and return the best score found.

    The mypyc flavour; mirrors _crown_walk_pure.

    Args:
        board: The flat board to walk, as `bytes` (see _bear_off_walk_native).
        home: The opponent home row (the crowning squares) as a bitmask.
        rays: The player's flat single-move ray table (see _flat_rays).
        slot: Which ray to follow, as anchor * 2 + direction.
        best: The best score found for this single so far (_NO_PATH if none).
        steps: Steps taken so far on the current path.

    Returns:
        The better of `best` and the best score reachable from this leg, a score
        being SINGLES_PATHS_MAX minus the path's step count.
    """
    k: i64 = slot * RAY_STRIDE
    while True:
        idx: i64 = rays[k]
        if idx == RAY_END:
            break
        k += 1
        if board[idx] != 0:
            break

        # Keep the shortest path to a crowning square found so far for this single
        # (converted into a score out of SINGLES_PATHS_MAX)
        if (home >> idx) & 1 and SINGLES_PATHS_MAX - steps > best:
            best = SINGLES_PATHS_MAX - steps

        # Change direction (add 1 to steps), unless the bound rules the leg out.
        if SINGLES_PATHS_MAX - steps - 1 > best:
            best = _crown_walk_native(
                board, home, rays, idx * 2 + (1 - (slot & 1)), best, steps + 1
            )

    return best


def _eval_paths_pure(state: list[int]) -> tuple[int, int, int]:
    """
    Score both players' bear-off and crowning prospects in one board pass.

    The pure-Python flavour, used when this module is interpreted.
    Must stay behaviour-identical to _eval_paths_native (see test_dual.py).

    One pass walks every piece of both colours as it is found, rather than
    bucketing the four piece kinds into lists and then making four driving
    calls over them: each piece's score is independent of the others, so the
    only thing the lists bought was the order the sums accumulate in.

    Args:
        state: The flat board to score.

    Returns:
        (doubles_path_score, singles_path_score, doubles_count), each as
        White's total minus Black's.
    """
    white_double_paths = 0
    black_double_paths = 0
    white_single_paths = 0
    black_single_paths = 0
    white_doubles = 0
    black_doubles = 0
    for idx in DARK_INDICES:
        code = state[idx]
        # A single walks towards the far home row (code 1 is White's single, 3
        # is Black's); a double walks towards its owner's own home row.
        if code == 1:
            best = _crown_walk_pure(
                state, HOME_INDICES_BLACK, CROWN_DIAGS_WHITE, idx, 0, _NO_PATH, 1
            )
            best = _crown_walk_pure(
                state, HOME_INDICES_BLACK, CROWN_DIAGS_WHITE, idx, 1, best, 1
            )
            if best != _NO_PATH:
                white_single_paths += best
        elif code == 2:
            white_doubles += 1
            best = _bear_off_walk_pure(
                state,
                1,
                HOME_INDICES_WHITE,
                BEAR_DIAGS_WHITE,
                idx,
                0,
                _NO_PATH,
                0,
                False,
                False,
            )
            best = _bear_off_walk_pure(
                state,
                1,
                HOME_INDICES_WHITE,
                BEAR_DIAGS_WHITE,
                idx,
                1,
                best,
                0,
                False,
                False,
            )
            if best != _NO_PATH:
                white_double_paths += best
        elif code == 3:
            best = _crown_walk_pure(
                state, HOME_INDICES_WHITE, CROWN_DIAGS_BLACK, idx, 0, _NO_PATH, 1
            )
            best = _crown_walk_pure(
                state, HOME_INDICES_WHITE, CROWN_DIAGS_BLACK, idx, 1, best, 1
            )
            if best != _NO_PATH:
                black_single_paths += best
        elif code == 4:
            black_doubles += 1
            best = _bear_off_walk_pure(
                state,
                3,
                HOME_INDICES_BLACK,
                BEAR_DIAGS_BLACK,
                idx,
                0,
                _NO_PATH,
                0,
                False,
                False,
            )
            best = _bear_off_walk_pure(
                state,
                3,
                HOME_INDICES_BLACK,
                BEAR_DIAGS_BLACK,
                idx,
                1,
                best,
                0,
                False,
                False,
            )
            if best != _NO_PATH:
                black_double_paths += best
    return (
        white_double_paths - black_double_paths,
        white_single_paths - black_single_paths,
        white_doubles - black_doubles,
    )


def _eval_paths_native(board: bytes) -> tuple[int, int, int]:
    """
    Score both players' bear-off and crowning prospects in one board pass.

    The mypyc flavour; mirrors _eval_paths_pure. Takes the board as `bytes` so
    that every square read, here and in the walkers this drives, is a native
    byte read (see _bear_off_walk_native).

    Args:
        board: The flat board to score, as `bytes`.

    Returns:
        (doubles_path_score, singles_path_score, doubles_count), each as
        White's total minus Black's.
    """
    white_double_paths: i64 = 0
    black_double_paths: i64 = 0
    white_single_paths: i64 = 0
    black_single_paths: i64 = 0
    white_doubles: i64 = 0
    black_doubles: i64 = 0
    k: i64 = 0
    while k < 32:
        idx: i64 = DARK_BYTES[k]
        k += 1
        code: i64 = board[idx]
        if code == 1:
            best = _crown_walk_native(
                board, HOME_MASK_BLACK, CROWN_RAYS_WHITE, idx * 2, _NO_PATH, 1
            )
            best = _crown_walk_native(
                board, HOME_MASK_BLACK, CROWN_RAYS_WHITE, idx * 2 + 1, best, 1
            )
            if best != _NO_PATH:
                white_single_paths += best
        elif code == 2:
            white_doubles += 1
            best = _bear_off_walk_native(
                board,
                1,
                HOME_MASK_WHITE,
                BEAR_RAYS_WHITE,
                idx * 2,
                _NO_PATH,
                0,
                False,
                False,
            )
            best = _bear_off_walk_native(
                board,
                1,
                HOME_MASK_WHITE,
                BEAR_RAYS_WHITE,
                idx * 2 + 1,
                best,
                0,
                False,
                False,
            )
            if best != _NO_PATH:
                white_double_paths += best
        elif code == 3:
            best = _crown_walk_native(
                board, HOME_MASK_WHITE, CROWN_RAYS_BLACK, idx * 2, _NO_PATH, 1
            )
            best = _crown_walk_native(
                board, HOME_MASK_WHITE, CROWN_RAYS_BLACK, idx * 2 + 1, best, 1
            )
            if best != _NO_PATH:
                black_single_paths += best
        elif code == 4:
            black_doubles += 1
            best = _bear_off_walk_native(
                board,
                3,
                HOME_MASK_BLACK,
                BEAR_RAYS_BLACK,
                idx * 2,
                _NO_PATH,
                0,
                False,
                False,
            )
            best = _bear_off_walk_native(
                board,
                3,
                HOME_MASK_BLACK,
                BEAR_RAYS_BLACK,
                idx * 2 + 1,
                best,
                0,
                False,
                False,
            )
            if best != _NO_PATH:
                black_double_paths += best
    return (
        int(white_double_paths - black_double_paths),
        int(white_single_paths - black_single_paths),
        int(white_doubles - black_doubles),
    )


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
    # Checker counts as two plain fields rather than a dict keyed by colour: the
    # search copies them on every node and reads them in every evaluation, where
    # a dict costs an allocation per copy and a tuple hash per read. The dict
    # form stays available through the checkers_total property below, which is
    # what the GUI, the save file and the tests use.
    checkers_white: int
    checkers_black: int
    winner: Color | None
    state_hash: int
    # Legal moves are generated lazily on first access: most search nodes are
    # eval-only leaves that never need them. None means "not yet computed".
    # _moves_list is the primary form; _all_legal_moves is derived from it and
    # only built when something outside the search asks for it.
    _moves_list: list[Move] | None
    _all_legal_moves: LegalMoves | None

    def __init__(
        self,
        state: State | LegacyState | None = None,
        turn: Color | None = None,
        checkers_total: dict[Color, int] | None = None,
        all_legal_moves: LegalMoves | None = None,
        winner: Color | None | object = DERIVE_WINNER,
        state_hash: int | None = None,
        clone_of: Position | None = None,
    ):
        # Fast path for copy(): every field is known, so skip make_position's
        # ladder of "derive this if it was not given" conditionals. One branch here
        # replaces about eight there, on a path the search takes once per node.
        if clone_of is not None:
            self.state = clone_of.state.copy()
            self.turn = clone_of.turn
            self._moves_list = None
            self._all_legal_moves = None
            self.checkers_white = clone_of.checkers_white
            self.checkers_black = clone_of.checkers_black
            self.winner = clone_of.winner
            self.state_hash = clone_of.state_hash
            return
        self.make_position(
            state, turn, checkers_total, all_legal_moves, winner, state_hash
        )

    def make_position(
        self,
        state: State | LegacyState | None = None,
        turn: Color | None = None,
        checkers_total: dict[Color, int] | None = None,
        all_legal_moves: LegalMoves | None = None,
        winner: Color | None | object = DERIVE_WINNER,
        state_hash: int | None = None,
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
        self.state = INITIAL_STATE_FLAT.copy() if state is None else _to_flat(state)
        self.turn = WHITE if turn is None else turn
        self._moves_list = None
        self._all_legal_moves = all_legal_moves
        if state is None:
            self.checkers_white = 12
            self.checkers_black = 12
        else:
            counts = checkers_total if checkers_total else self.count_checkers()
            self.checkers_white = counts[WHITE]
            self.checkers_black = counts[BLACK]
        self.winner = (
            None
            if state is None
            else winner  # type: ignore[assignment]
            if winner is not DERIVE_WINNER
            else WHITE
            if not self.checkers_white
            else BLACK
            if not self.checkers_black
            else None
        )
        self.state_hash = state_hash if state_hash else _make_state_hash(self.state)

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
    def all_legal_moves(self, value: LegalMoves | None) -> None:
        """Set the position's legal moves, or None to drop them."""
        self._all_legal_moves = value
        # Drop the flat form too; it is re-derived from the board on next access.
        self._moves_list = None

    @property
    def checkers_total(self) -> dict[Color, int]:
        """Each player's checker count, a double counting as two."""
        return {WHITE: self.checkers_white, BLACK: self.checkers_black}

    @checkers_total.setter
    def checkers_total(self, value: dict[Color, int]) -> None:
        """Set both players' checker counts from the dict form."""
        self.checkers_white = value[WHITE]
        self.checkers_black = value[BLACK]

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

    def piece_at(self, cell: Cell) -> Piece | None:
        """
        Return the piece occupying a cell.

        Args:
            cell: The board cell to read.

        Returns:
            The piece as a (color, type) pair, or None if the cell is empty.
        """
        return CODE_TO_PIECE[self.state[_index(cell)]]

    # Impasse game functions

    def _append_slides(
        self,
        origin: Cell,
        origin_index: int,
        origin_code: int,
        turn_home: bytes,
        far_home: bytes,
        out: list[Move],
    ) -> None:
        """
        Append every slide available to the checker on a cell.

        Appends rather than returning a dict: generation runs on every internal
        search node and the per-origin dicts were allocated only to be flattened.
        Everything that depends on the position rather than on the origin is
        passed in: the caller has read the square already, and the two home-row
        tables are the same for every origin it visits.

        Args:
            origin: The cell of the moving checker.
            origin_index: That cell's flat board index.
            origin_code: The square code there, which must be a checker of the
                player to move.
            turn_home: The mover's home-row table (HOME_TABLE).
            far_home: The opponent's home-row table.
            out: The move list to append to. Tags are "S" for a regular slide,
                "SB" for a slide onto the player's home row (a bear off), or "SC"
                for a slide of a single onto the far row (enabling a crowning).
        """
        state = self.state
        rays = SLIDE_RAYS[origin_code]
        origin_is_single = origin_code == 1 or origin_code == 3
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

    def _append_transposes(
        self,
        origin: Cell,
        origin_index: int,
        origin_code: int,
        single: int,
        turn_home: bytes,
        far_home: bytes,
        out: list[Move],
    ) -> None:
        """
        Append every transpose available to the crown on a cell.

        A transpose swaps with a single on the *adjacent* square in one of the
        crown's two move directions, which is the first entry of that
        direction's slide ray -- empty exactly when the square is off the board.
        So this reads SLIDE_RAYS too, instead of stepping a direction vector and
        bounds-checking the result.

        Args:
            origin: The cell of the moving crown.
            origin_index: That cell's flat board index.
            origin_code: The square code there, which must be a crown of the
                player to move.
            single: The mover's single square code.
            turn_home: The mover's home-row table (HOME_TABLE).
            far_home: The opponent's home-row table.
            out: The move list to append to. Tags are "T" for a regular transpose,
                "TB" for one onto the player's home row (a bear off), or "TC" for
                one leaving a single on the far row (enabling a crowning).
        """
        state = self.state
        rays = SLIDE_RAYS[origin_code]
        origin_on_far_home = far_home[origin_index] == 1
        base = origin_index * 2
        for i in (0, 1):
            ray = rays[base + i]
            if not ray:
                continue
            candidate, candidate_index = ray[0]
            if state[candidate_index] == single:
                # Transpose leading to bear off
                if turn_home[candidate_index]:
                    out.append((origin, candidate, "TB"))
                # Transpose leading to available crowning
                elif origin_on_far_home:
                    out.append((origin, candidate, "TC"))
                # Regular transpose
                else:
                    out.append((origin, candidate, "T"))

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
        single = SINGLE_CODE[turn]
        double = single + 1
        state = self.state
        turn_home = HOME_TABLE[turn]
        far_home = HOME_TABLE[OPPOSITE_COLOR[turn]]
        # Scanned by index, with the cell read off CELL_AT_INDEX only for the
        # mover's own squares: the two tests that find those are what every one
        # of the 32 squares pays for.
        for idx in DARK_INDICES:
            code = state[idx]
            if code == single:
                # Singles slide.
                self._append_slides(
                    CELL_AT_INDEX[idx], idx, code, turn_home, far_home, out
                )
            elif code == double:
                # Crowns transpose as well as slide; transposes come first, as
                # they did when this merged two dicts.
                cell = CELL_AT_INDEX[idx]
                self._append_transposes(
                    cell, idx, code, single, turn_home, far_home, out
                )
                self._append_slides(cell, idx, code, turn_home, far_home, out)
        # Impasse
        if not out:
            return [
                (CELL_AT_INDEX[idx], None, "B")
                for idx in DARK_INDICES
                if state[idx] == single or state[idx] == double
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
        _, origin_code, target_index, target_code = self._move_squares(
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

    def _bear_off_one(self) -> int:
        """
        Take one checker of the player to move off the board.

        Returns:
            How many checkers that player has left.
        """
        if self.turn == WHITE:
            self.checkers_white -= 1
            return self.checkers_white
        self.checkers_black -= 1
        return self.checkers_black

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
            remaining = self._bear_off_one()
            # Check for win
            if not remaining:
                self.winner = self.turn
                # A decided position has no moves to offer.
                self._moves_list = []
                self._all_legal_moves = None
            # Might make crowning possible
            else:
                self.check_for_crownings_and_change_turn()
        # Slide + Bear off
        elif tag == "SB":
            self._bear_off_one()
            # Might make crowning possible
            self.check_for_crownings_and_change_turn()
        # Transpose + Bear off. Leaves a new single on the origin square, which is the
        # rules' "come to have another on-board single" crowning trigger.
        elif tag == "TB":
            self._bear_off_one()
            self.check_for_crownings_and_change_turn()
        # Potential crowning
        elif tag in ("SC", "TC") or tag == "C":
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
        # One pass over the board scores every piece of both colours. The native
        # flavour reads a `bytes` snapshot: taking it costs ~0.2 us, and the
        # walkers it drives read the board hundreds of times per evaluation,
        # where a `bytes` index is a native byte read and a `list[int]` index
        # has to unbox an object.
        if NATIVE:
            doubles_path_score, singles_path_score, doubles_count = _eval_paths_native(
                bytes(self.state)
            )
        else:
            doubles_path_score, singles_path_score, doubles_count = _eval_paths_pure(
                self.state
            )
        checkers_count = self.checkers_black - self.checkers_white
        if checkers_count:
            value = CHECKERS_COUNT_WEIGHT * checkers_count
        else:
            value = DOUBLES_WEIGHT * doubles_count
        value += (
            DOUBLES_PATHS_WEIGHT * doubles_path_score
            + SINGLES_PATHS_WEIGHT * singles_path_score
        )
        return value
