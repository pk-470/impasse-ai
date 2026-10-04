"""
Golden-master behaviour lock for the performance work.

These digests are computed from the public, cell-based API only, so they must
stay identical across every behaviour-preserving change. Any change here means
an optimization altered results.
"""

from collections.abc import Callable

import pytest

from tests.golden_util import (
    ordering_digest,
    search_results,
    search_values_legal_tied,
    trajectory_digest,
)

# Captured from the post-Phase-1 engine (the correct baseline).
TRAJECTORY_DIGEST = "54e32c7aec118d002c3158dd0dd2711cc7a6fbabda9e0d3d440a20176c5b1f81"
# Move ordering, which TRAJECTORY_DIGEST is blind to because it sorts.
ORDERING_DIGEST = "77671a37bb958cd34509f2e49598ffdcefd47b580c5c704ff2d90108b7fd28ea"
SEARCH_RESULTS = (
    (0, 0, ((2, 6), (1, 5), "S")),
    (6, -10, ((7, 5), (4, 2), "S")),
    (12, -32, ((4, 2), (3, 1), "T")),
    (20, 121, ((4, 4), (2, 0), "C")),
    (30, -64, ((4, 0), (5, 7), "C")),
)


class TestDigests:
    @pytest.mark.parametrize(
        "digest_fn,positions,expected",
        [
            (trajectory_digest, 4000, TRAJECTORY_DIGEST),
            (ordering_digest, 1200, ORDERING_DIGEST),
        ],
    )
    def test_digest_unchanged(
        self,
        digest_fn: Callable[[], tuple[str, int]],
        positions: int,
        expected: str,
    ) -> None:
        digest, n = digest_fn()
        assert n == positions
        assert digest == expected


class TestSearchResults:
    def test_exact_moves_unchanged(self) -> None:
        assert search_results() == SEARCH_RESULTS

    def test_values_hold_with_a_legal_tied_move(self) -> None:
        """
        Survives a move-ordering change, which the exact-move golden above does
        not: it pins each search value and only requires the chosen move to be
        legal and to attain it.
        """
        expected_values = tuple((off, value) for off, value, _ in SEARCH_RESULTS)
        results = search_values_legal_tied()
        assert tuple((off, value) for off, value, _, _ in results) == expected_values
        for off, _, legal, tied in results:
            assert legal, f"offset {off}: returned move is not legal"
            assert tied, f"offset {off}: returned move does not attain the search value"
