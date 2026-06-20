"""Argument-parsing tests for the impasse command-line entry point."""

import pytest

from impasse.__main__ import build_parser


def test_dev_flag_defaults_false():
    assert build_parser().parse_args([]).dev is False


def test_dev_flag_combines_with_ai():
    args = build_parser().parse_args(["--ai", "white", "--dev"])
    assert args.dev is True
    assert args.ai == "white"


def test_dev_flag_allowed_without_ai():
    # --dev is independent of the --ai/--time mode group.
    assert build_parser().parse_args(["--dev"]).dev is True


def test_ai_and_time_remain_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--ai", "white", "--time", "60"])
