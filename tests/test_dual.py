"""The two pathfinder flavours must agree exactly.

`position.py` carries a pure-Python and a mypyc-native implementation of each
pathfinder and picks one via `NATIVE` (see position.NATIVE). Only one of them
runs in any given build, so a divergence would otherwise show up as a silent
evaluation change the moment the engine is compiled — or stop being caught the
moment it is. These tests run *both*, in every build, over positions reached by
real play, and assert they are identical.
"""

import random

from impasse.position import (
    BEAR_DIAGS,
    BEAR_RAYS,
    BLACK,
    CROWN_DIAGS,
    CROWN_RAYS,
    HOME_INDICES,
    HOME_MASK,
    WHITE,
    Move,
    Position,
    _eval_paths_native,
    _eval_paths_pure,
)


def _selfplay_positions(count, seed=12345):
    """Positions from a seeded random game, so the boards are reachable ones."""
    rng = random.Random(seed)
    out = []
    pos = Position()
    while len(out) < count:
        if pos.winner is not None:
            pos = Position()
            continue
        out.append(pos)
        moves: list[Move] = [
            (origin, target, tag)
            for origin, targets in pos.all_legal_moves.items()
            for target, tag in targets.items()
        ]
        if not moves:
            pos = Position()
            continue
        pos = pos.new_position_after_move(*rng.choice(moves))
    return out


def test_eval_paths_flavours_agree():
    """Both evaluation passes return the same three scores on every board.

    This covers the walkers and the board pass that drives them: the pure
    flavour reads the `list[int]` board, the native one a `bytes` snapshot of
    it, and only one of the two runs in any given build.
    """
    checked = 0
    for pos in _selfplay_positions(400):
        pure = _eval_paths_pure(pos.state)
        native = _eval_paths_native(bytes(pos.state))
        assert pure == native, (
            f"evaluation flavours disagree: pure={pure} native={native} "
            f"state={pos.state}"
        )
        checked += 1
    assert checked == 400


def test_split_colour_tables_match_the_dicts():
    """The per-colour globals the evaluation pass reads are the dict entries.

    The fused pass indexes these instead of hashing a Color per piece, so a
    mismatch would silently score one side with the other's tables.
    """
    from impasse import position as P

    for color, suffix in ((WHITE, "WHITE"), (BLACK, "BLACK")):
        for name in ("BEAR_DIAGS", "CROWN_DIAGS", "HOME_INDICES",
                     "BEAR_RAYS", "CROWN_RAYS", "HOME_MASK"):
            assert getattr(P, f"{name}_{suffix}") == getattr(P, name)[color], (
                f"{name}_{suffix} does not match {name}[{color}]"
            )
    assert list(P.DARK_BYTES) == P.DARK_INDICES


def test_home_mask_matches_home_indices():
    """The bitmask home row encodes exactly the frozenset one.

    HOME_MASK is stored two's-complement because square 63 is a home square and
    an unsigned mask overflows mypyc's signed i64; this pins the equivalence.
    """
    for color in (WHITE, BLACK):
        mask = HOME_MASK[color]
        for idx in range(64):
            assert bool((mask >> idx) & 1) == (idx in HOME_INDICES[color]), (
                f"home mask/set mismatch at {idx} for {color}"
            )


def test_rays_match_nested_diagonals():
    """The flat `bytes` ray table holds the same indices as the nested lists.

    Each (anchor, direction) slot owns a fixed-stride span terminated by
    RAY_END, which is what lets the native walkers read a step as one byte and
    stop without a length. Both the squares and the terminator are pinned.
    """
    from impasse.position import RAY_END, RAY_STRIDE

    for color in (WHITE, BLACK):
        for flat, nested in (
            (BEAR_RAYS[color], BEAR_DIAGS[color]),
            (CROWN_RAYS[color], CROWN_DIAGS[color]),
        ):
            assert len(flat) == 128 * RAY_STRIDE
            for anchor in range(64):
                for i in (0, 1):
                    base = (anchor * 2 + i) * RAY_STRIDE
                    span = flat[base : base + RAY_STRIDE]
                    ray = nested[anchor][i]
                    assert list(span[: len(ray)]) == ray
                    assert span[len(ray)] == RAY_END


def test_home_table_matches_home_indices():
    """The 0/1 byte home table encodes exactly the frozenset one."""
    from impasse.position import HOME_TABLE

    for color in (WHITE, BLACK):
        table = HOME_TABLE[color]
        assert len(table) == 64
        for idx in range(64):
            assert bool(table[idx]) == (idx in HOME_INDICES[color]), (
                f"home table/set mismatch at {idx} for {color}"
            )


def test_slide_rays_match_diagonals():
    """SLIDE_RAYS carries the right (cell, index) pairs for every piece code."""
    from impasse.position import (
        DIAG_INDICES,
        MOVE_DIRECTIONS,
        PIECE_TO_CODE,
        SLIDE_RAYS,
        _index,
    )

    for piece, dirs in MOVE_DIRECTIONS.items():
        rays = SLIDE_RAYS[PIECE_TO_CODE[piece]]
        for anchor in range(64):
            for i in (0, 1):
                expected = DIAG_INDICES.get((anchor, dirs[i]), [])
                got = rays[anchor * 2 + i]
                assert [idx for _, idx in got] == expected
                # The paired cell must be the one that index decodes to.
                for cell, idx in got:
                    assert _index(cell) == idx


def test_turn_bit_is_keyed_by_value_not_identity():
    """An unpickled turn colour must still map to the right side-to-move bit.

    gui.py's undo path assigns a turn colour straight out of a pickle, which is an
    equal but distinct tuple. The transposition-table key is built from TURN_BIT,
    so an identity-based lookup there would silently key every entry as Black.
    """
    import pickle

    from impasse.position import TURN_BIT

    for color, expected in ((WHITE, 0), (BLACK, 1)):
        revived = pickle.loads(pickle.dumps(color))
        assert revived is not color, "pickle unexpectedly interned the tuple"
        assert TURN_BIT[revived] == expected


def test_diag_rays_match_diag_indices():
    """DIAG_RAYS holds the same indices as the (anchor, direction)-keyed table.

    This is the flat table ai.ordered_moves reads when scoring how well a slide
    blocks an enemy checker, and it was the one new table not pinned here.
    """
    from impasse.position import DIAG_INDICES, DIAG_RAYS, MOVE_DIRECTIONS

    for piece, dirs in MOVE_DIRECTIONS.items():
        rays = DIAG_RAYS[piece]
        assert len(rays) == 128
        for anchor in range(64):
            for i in (0, 1):
                assert rays[anchor * 2 + i] == DIAG_INDICES.get(
                    (anchor, dirs[i]), []
                ), f"DIAG_RAYS mismatch for {piece} at anchor {anchor} dir {i}"
