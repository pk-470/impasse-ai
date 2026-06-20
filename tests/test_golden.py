"""Golden-master behaviour lock for the performance refactor.

These digests are computed from the public, cell-based API only, so they MUST stay
identical across every behaviour-preserving Phase 2 change (including the flat
int-board rewrite). Any change here means the optimization altered results.
"""

from tests.golden_util import (
    search_results,
    search_values_legal_tied,
    trajectory_digest,
)

# Captured from the post-Phase-1 engine (the correct baseline).
TRAJECTORY_DIGEST = "54e32c7aec118d002c3158dd0dd2711cc7a6fbabda9e0d3d440a20176c5b1f81"
SEARCH_RESULTS = (
    (0, 0, ((2, 6), (1, 5), "S")),
    (6, -10, ((7, 5), (4, 2), "S")),
    (12, -32, ((4, 2), (3, 1), "T")),
    (20, 121, ((4, 4), (2, 0), "C")),
    (30, -64, ((4, 0), (5, 7), "C")),
)


def test_trajectory_digest_unchanged():
    digest, n = trajectory_digest()
    assert n == 4000
    assert digest == TRAJECTORY_DIGEST


def test_search_results_unchanged():
    assert search_results() == SEARCH_RESULTS


def test_search_result_values_legal_and_tied():
    """Value-only golden, robust to move-ordering changes.

    The exact-move golden above legitimately breaks when a search-quality change
    (killers, history, aspiration, TT replacement) picks a different equally-good
    move. This one survives that: it pins each search VALUE, and only requires the
    chosen move to be legal and to attain that value.
    """
    expected_values = tuple((off, value) for off, value, _ in SEARCH_RESULTS)
    results = search_values_legal_tied()
    assert tuple((off, value) for off, value, _, _ in results) == expected_values
    for off, _, legal, tied in results:
        assert legal, f"offset {off}: returned move is not legal"
        assert tied, f"offset {off}: returned move does not attain the search value"
