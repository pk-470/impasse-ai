"""The Impasse board: state, move generation, move application and evaluation."""

from __future__ import annotations

import random
from typing import Final, Literal

from mypy_extensions import i64, mypyc_attr

type Cell = tuple[int, int]
type Color = tuple[int, int, int]
type Piece = tuple[Color, int]
type State = list[int]
type MoveTag = Literal["S", "SB", "SC", "T", "TB", "TC", "C", "B"]
type MoveDestination = Cell | None
type MoveDict = dict[MoveDestination, MoveTag]
type LegalMoves = dict[Cell, MoveDict]
# A single move: where from, where to (None for a bear off) and what kind.
type Move = tuple[Cell, MoveDestination, MoveTag]

WHITE: Final[Color] = (255, 255, 255)
BLACK: Final[Color] = (0, 0, 0)
OPPOSITE_COLOR: Final[dict[Color, Color]] = {WHITE: BLACK, BLACK: WHITE}

EMPTY: Final = 0
PIECE_TO_CODE: Final[dict[Piece | None, int]] = {
    None: EMPTY,
    (WHITE, 1): 1,
    (WHITE, 2): 2,
    (BLACK, 1): 3,
    (BLACK, 2): 4,
}
CODE_TO_PIECE: Final[dict[int, Piece | None]] = {c: p for p, c in PIECE_TO_CODE.items()}
# 1 single, 2 crown; index 0 (empty) has no type.
CODE_TYPE: Final[bytes] = bytes((0, 1, 2, 1, 2))
SINGLE_CODE: Final[dict[Color, int]] = {WHITE: 1, BLACK: 3}
COLOR_CODES: Final[dict[Color, tuple[int, int]]] = {WHITE: (1, 2), BLACK: (3, 4)}


def _index(cell: Cell) -> int:
    """Return the flat board index (row * 8 + col) of a cell."""
    return cell[0] * 8 + cell[1]


# The 32 dark squares in board order, and their flat indices.
DARK_CELLS: Final[list[Cell]] = [
    (i, j) for i in range(8) for j in range(8) if (i + j) % 2 == 0
]
DARK_INDICES: Final[list[int]] = [_index(cell) for cell in DARK_CELLS]
CELL_AT_INDEX: Final[list[Cell]] = [(idx // 8, idx % 8) for idx in range(64)]
for _cell in DARK_CELLS:
    CELL_AT_INDEX[_index(_cell)] = _cell

# The starting layout, as the cells each square code occupies.
_INITIAL_LAYOUT: Final[dict[int, tuple[Cell, ...]]] = {
    1: ((0, 0), (3, 1), (4, 0), (7, 1)),
    2: ((1, 7), (2, 6), (5, 7), (6, 6)),
    3: ((0, 6), (3, 7), (4, 6), (7, 7)),
    4: ((1, 1), (2, 0), (5, 1), (6, 0)),
}
INITIAL_STATE: Final[list[int]] = [EMPTY] * 64
for _code, _cells in _INITIAL_LAYOUT.items():
    for _cell in _cells:
        INITIAL_STATE[_index(_cell)] = _code


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
    for i, j in DARK_CELLS
    for checker in MOVE_DIRECTIONS
    for d in MOVE_DIRECTIONS[checker]
}
# The same diagonals as flat indices, keyed by (origin index, direction).
DIAG_INDICES: Final[dict[tuple[int, Cell], list[int]]] = {
    (_index(cell), d): [_index(c) for c in cells]
    for (cell, d), cells in DIAGONALS.items()
}
# The same diagonals indexed by [origin index * 2 + direction choice].
DIAG_RAYS: Final[dict[Piece, list[list[int]]]] = {
    piece: [
        DIAG_INDICES.get((anchor, MOVE_DIRECTIONS[piece][i]), [])
        for anchor in range(64)
        for i in (0, 1)
    ]
    for piece in MOVE_DIRECTIONS
}
# As (cell, index) pairs.
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
# Home rows as 64-byte 0/1 tables.
HOME_TABLE: Final[dict[Color, bytes]] = {
    color: bytes(1 if idx in indices else 0 for idx in range(64))
    for color, indices in HOME_INDICES.items()
}
# Side to move as a bit, for the transposition-table key.
TURN_BIT: Final[dict[Color, int]] = {WHITE: 0, BLACK: 1}
# The same diagonals indexed by [anchor][direction choice]. BEAR_DIAGS follows
# a double's move directions, CROWN_DIAGS a single's.
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

# True when mypyc compiled this module; selects the _pure or _native flavour.
NATIVE: Final[bool] = not __file__.endswith((".py", ".pyc"))


def _signed64(mask: int) -> int:
    """Reinterpret a 64-bit mask as signed; bit 63 overflows mypyc's i64."""
    return mask - (1 << 64) if mask >= (1 << 63) else mask


HOME_MASK: Final[dict[Color, int]] = {
    color: _signed64(sum(1 << idx for idx in indices))
    for color, indices in HOME_INDICES.items()
}
# Fixed-stride spans, one per (anchor, direction) slot, each ending in RAY_END.
RAY_STRIDE: Final = 8
RAY_END: Final = 255


def _flat_rays(diags: list[list[list[int]]]) -> bytes:
    """Pack per-anchor diagonals into fixed-stride spans, each ending in RAY_END."""
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
DARK_BYTES: Final = bytes(DARK_INDICES)

# Sentinel "no path found" score, below any score a real path can reach.
_NO_PATH: Final = -1_000

# Sentinel for `winner`: read it off the board. None means "nobody has won".
DERIVE_WINNER: Final[object] = object()

DOUBLES_PATHS_MAX: Final = 10
SINGLES_PATHS_MAX: Final = 10

CHECKERS_COUNT_WEIGHT: Final = 120
DOUBLES_PATHS_WEIGHT: Final = 8
SINGLES_PATHS_WEIGHT: Final = 2
DOUBLES_WEIGHT: Final = 1
# Score of a decided position, far above any heuristic score.
WIN_VALUE: Final = 100_000


# A random id per (square index, code), 61 bits wide.
ZOBRIST_BITS: Final = 61
random.seed(42)
rand_ids: Final[list[list[int]]] = [[0] * 5 for _ in range(64)]
for _idx in DARK_INDICES:
    for _code in range(5):
        rand_ids[_idx][_code] = random.getrandbits(ZOBRIST_BITS)


def _moves_to_dict(moves: list[Move]) -> LegalMoves:
    """Group a flat move list into the {origin: {target: tag}} form."""
    grouped: LegalMoves = {}
    for origin, target, tag in moves:
        entry = grouped.get(origin)
        if entry is None:
            grouped[origin] = {target: tag}
        else:
            entry[target] = tag
    return grouped


def _make_state_hash(state: list[int]) -> int:
    """Compute a board's Zobrist hash from scratch."""
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
    """Best bear-off score for one double's paths; mirrors the native flavour."""
    for idx in diags[anchor][i]:
        piece = state[idx]
        if piece == EMPTY:
            # Stepping from a single onto an empty cell costs a step.
            if not prev_empty:
                steps += 1
                prev_empty = True
            changed_dir = False
            # Add one step before changing direction
            new_steps = steps + 1
        elif piece == single:
            # Landing on a single costs a step, unless the direction change paid.
            if not (changed_dir and prev_empty):
                steps += 1
            changed_dir = False
            prev_empty = False
            # Change direction without adding step
            new_steps = steps
        else:
            break

        # Keep the shortest bear-off path found, scored out of DOUBLES_PATHS_MAX.
        if idx in home and DOUBLES_PATHS_MAX - steps > best:
            best = DOUBLES_PATHS_MAX - steps
            break

        # Steps only grow, so a leg that cannot beat `best` is dead.
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
    """Best crowning score for one single's paths; mirrors the native flavour."""
    for idx in diags[anchor][i]:
        if state[idx] != EMPTY:
            break

        # Keep the shortest path to a crowning square, scored out of SINGLES_PATHS_MAX.
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
    """Best bear-off score for one double's paths; the mypyc i64/bytes flavour."""
    k: i64 = slot * RAY_STRIDE
    while True:
        idx: i64 = rays[k]
        if idx == RAY_END:
            break
        k += 1
        piece: i64 = board[idx]
        new_steps: i64 = 0
        if piece == 0:
            # Stepping from a single onto an empty cell costs a step.
            if not prev_empty:
                steps += 1
                prev_empty = True
            changed_dir = False
            # Add one step before changing direction
            new_steps = steps + 1
        elif piece == single:
            # Landing on a single costs a step, unless the direction change paid.
            if not (changed_dir and prev_empty):
                steps += 1
            changed_dir = False
            prev_empty = False
            # Change direction without adding step
            new_steps = steps
        else:
            break

        # Keep the shortest bear-off path found, scored out of DOUBLES_PATHS_MAX.
        if (home >> idx) & 1 and DOUBLES_PATHS_MAX - steps > best:
            best = DOUBLES_PATHS_MAX - steps
            break

        # Steps only grow, so a leg that cannot beat `best` is dead.
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
    """Best crowning score for one single's paths; the mypyc i64/bytes flavour."""
    k: i64 = slot * RAY_STRIDE
    while True:
        idx: i64 = rays[k]
        if idx == RAY_END:
            break
        k += 1
        if board[idx] != 0:
            break

        # Keep the shortest path to a crowning square, scored out of SINGLES_PATHS_MAX.
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
    Score both players' prospects in one pass, as White's total minus Black's:
    (doubles_path_score, singles_path_score, doubles_count).
    """
    white_double_paths = 0
    black_double_paths = 0
    white_single_paths = 0
    black_single_paths = 0
    white_doubles = 0
    black_doubles = 0
    for idx in DARK_INDICES:
        code = state[idx]
        # A single walks towards the far home row, a double towards its own.
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
    """The mypyc flavour of _eval_paths_pure, reading the board as `bytes`."""
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

    state: State
    turn: Color
    # The dict form is the checkers_total property below.
    checkers_white: int
    checkers_black: int
    winner: Color | None
    state_hash: int
    # Generated lazily; None means "not yet computed".
    _moves_list: list[Move] | None
    _all_legal_moves: LegalMoves | None

    def __init__(
        self,
        state: State | None = None,
        turn: Color | None = None,
        checkers_total: dict[Color, int] | None = None,
        all_legal_moves: LegalMoves | None = None,
        winner: Color | None | object = DERIVE_WINNER,
        state_hash: int | None = None,
        clone_of: Position | None = None,
    ):
        # Fast path for copy(): every field is known.
        """Create a position, or clone `clone_of` when it is given."""
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
        state: State | None = None,
        turn: Color | None = None,
        checkers_total: dict[Color, int] | None = None,
        all_legal_moves: LegalMoves | None = None,
        winner: Color | None | object = DERIVE_WINNER,
        state_hash: int | None = None,
    ) -> None:
        """
        Set up the position from the given data, deriving whatever is omitted.

        Args:
            state: The board as 64 square codes, which the position takes
                ownership of; a fresh starting board if None.
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
        self.state = INITIAL_STATE.copy() if state is None else state
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
        # Drop the flat form too; it is re-derived on next access.
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
        # Legal moves are left unset; the next access re-derives them.
        return Position(clone_of=self)

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
        Append the checker's slides to `out`, tagged "SB" (bear off), "SC"
        (enables a crowning) or "S".
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
        Append the crown's transposes to `out`, tagged "TB" (bear off), "TC"
        (enables a crowning) or "T".
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
        # No mover's single on the far row means no crowning is owed.
        owed = False
        for target in far_row:
            if state[target[0] * 8 + target[1]] == single:
                owed = True
                break
        if not owed:
            return []

        out: list[Move] = []
        # One helper can crown several far-row singles.
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
        for idx in DARK_INDICES:
            code = state[idx]
            if code == single:
                self._append_slides(
                    CELL_AT_INDEX[idx], idx, code, turn_home, far_home, out
                )
            elif code == double:
                # Crowns transpose as well as slide; transposes come first.
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
        The squares a move changes, as (origin_index, origin_code, target_index,
        target_code); target_index is -1 when only the origin square changes.
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
        """Take one of the mover's checkers off the board; return how many remain."""
        if self.turn == WHITE:
            self.checkers_white -= 1
            return self.checkers_white
        self.checkers_black -= 1
        return self.checkers_black

    def _advance(self, tag: MoveTag) -> None:
        """Advance turn, winner and pending crownings once a move's squares are set."""
        # Bear off
        if tag == "B":
            remaining = self._bear_off_one()
            if not remaining:
                self.winner = self.turn
                # A decided position has no moves to offer.
                self._moves_list = []
                self._all_legal_moves = None
            else:
                self.check_for_crownings_and_change_turn()
        # Slide or transpose onto the home row. A "TB" leaves a new single on the
        # origin, which can itself owe a crowning.
        elif tag in ("SB", "TB"):
            self._bear_off_one()
            self.check_for_crownings_and_change_turn()
        # Crowning, or a move that enables one
        elif tag in ("SC", "TC", "C"):
            self.check_for_crownings_and_change_turn()
        # Regular slide or transpose
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
