"""
The two pathfinder flavours must agree exactly.

`position.py` carries a pure-Python and a mypyc-native implementation of each
pathfinder and picks one via `NATIVE` (see position.NATIVE). Only one of them
runs in any given build, so a divergence would otherwise show up as a silent
evaluation change the moment the engine is compiled — or stop being caught the
moment it is. These tests run *both*, in every build, over positions reached by
real play, and assert they are identical.
"""

import random

import pytest

from impasse.position import (
    BLACK,
    HOME_INDICES,
    WHITE,
    Move,
    Position,
    _eval_paths_native,
    _eval_paths_pure,
)


def _selfplay_positions(count: int, seed: int = 12345) -> list[Position]:
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


class TestEvalPathFlavours:
    def test_both_flavours_agree_on_every_board(self) -> None:
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


class TestHomeRowTables:
    @pytest.mark.parametrize(
        "table,holds", [("HOME_MASK", "bit"), ("HOME_TABLE", "byte")]
    )
    def test_encode_the_same_squares_as_home_indices(
        self, table: str, holds: str
    ) -> None:
        from impasse import position as P

        for color in (WHITE, BLACK):
            encoded = getattr(P, table)[color]
            if holds == "byte":
                assert len(encoded) == 64
            for idx in range(64):
                present = (
                    bool((encoded >> idx) & 1) if holds == "bit" else bool(encoded[idx])
                )
                assert present == (idx in HOME_INDICES[color]), (
                    f"{table} mismatch at {idx} for {color}"
                )


class TestRayTables:
    @pytest.mark.parametrize(
        "flat,nested", [("BEAR_RAYS", "BEAR_DIAGS"), ("CROWN_RAYS", "CROWN_DIAGS")]
    )
    def test_flat_rays_match_the_nested_diagonals(
        self, flat: str, nested: str
    ) -> None:
        from impasse import position as P
        from impasse.position import RAY_END, RAY_STRIDE

        for color in (WHITE, BLACK):
            spans, lists = getattr(P, flat)[color], getattr(P, nested)[color]
            assert len(spans) == 128 * RAY_STRIDE
            for anchor in range(64):
                for i in (0, 1):
                    base = (anchor * 2 + i) * RAY_STRIDE
                    span = spans[base : base + RAY_STRIDE]
                    ray = lists[anchor][i]
                    assert list(span[: len(ray)]) == ray
                    assert span[len(ray)] == RAY_END

    def test_slide_rays_carry_matching_cell_index_pairs(self) -> None:
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
                    got = rays[anchor * 2 + i]
                    assert [idx for _, idx in got] == DIAG_INDICES.get(
                        (anchor, dirs[i]), []
                    )
                    for cell, idx in got:
                        assert _index(cell) == idx

    def test_diag_rays_match_diag_indices(self) -> None:
        from impasse.position import DIAG_INDICES, DIAG_RAYS, MOVE_DIRECTIONS

        for piece, dirs in MOVE_DIRECTIONS.items():
            rays = DIAG_RAYS[piece]
            assert len(rays) == 128
            for anchor in range(64):
                for i in (0, 1):
                    assert rays[anchor * 2 + i] == DIAG_INDICES.get(
                        (anchor, dirs[i]), []
                    ), f"DIAG_RAYS mismatch for {piece} at anchor {anchor} dir {i}"


class TestPerColourTables:
    def test_split_globals_match_the_dict_entries(self) -> None:
        from impasse import position as P

        for color, suffix in ((WHITE, "WHITE"), (BLACK, "BLACK")):
            for name in (
                "BEAR_DIAGS",
                "CROWN_DIAGS",
                "HOME_INDICES",
                "BEAR_RAYS",
                "CROWN_RAYS",
                "HOME_MASK",
            ):
                assert getattr(P, f"{name}_{suffix}") == getattr(P, name)[color], (
                    f"{name}_{suffix} does not match {name}[{color}]"
                )
        assert list(P.DARK_BYTES) == P.DARK_INDICES


class TestTurnBit:
    def test_is_keyed_by_value_not_identity(self) -> None:
        import pickle

        from impasse.position import TURN_BIT

        for color, expected in ((WHITE, 0), (BLACK, 1)):
            revived = pickle.loads(pickle.dumps(color))
            assert revived is not color, "pickle unexpectedly interned the tuple"
            assert TURN_BIT[revived] == expected
