"""Argument-parsing tests for the impasse command-line entry point."""

import pytest

from impasse.__main__ import build_parser


class TestBuildParser:
    @pytest.mark.parametrize(
        "argv,dev,ai",
        [
            ([], False, None),
            (["--dev"], True, None),
            (["--ai", "white", "--dev"], True, "white"),
            (["--ai", "black"], False, "black"),
        ],
    )
    def test_dev_flag_is_independent_of_the_mode_group(
        self, argv: list[str], dev: bool, ai: str | None
    ) -> None:
        args = build_parser().parse_args(argv)
        assert args.dev is dev
        assert args.ai == ai

    def test_ai_and_time_are_mutually_exclusive(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--ai", "white", "--time", "60"])

    @pytest.mark.parametrize("seconds", ["0", "-1", "abc"])
    def test_time_rejects_non_positive_seconds(self, seconds: str) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--time", seconds])
