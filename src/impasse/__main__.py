"""Command-line entry point. Launch with `impasse` or `python -m impasse`."""

import argparse

from impasse.play import play
from impasse.position import BLACK, WHITE

_AI_COLORS = {"white": WHITE, "black": BLACK}


def _positive_int(value: str) -> int:
    seconds = int(value)
    if seconds <= 0:
        raise argparse.ArgumentTypeError("time must be a positive number of seconds")
    return seconds


def build_parser():
    parser = argparse.ArgumentParser(
        prog="impasse",
        description=(
            "Play Impasse, Mark Steere's board game, against a human or an "
            "alpha-beta search AI."
        ),
        epilog=(
            "In-game keys: 'c' toggles cell names, 'z' undoes the last move, "
            "'n' starts a new game."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--ai",
        choices=("white", "black"),
        metavar="COLOR",
        help=(
            "let the AI play this colour ('white' moves first); omit for a "
            "human-vs-human game"
        ),
    )
    mode.add_argument(
        "--time",
        type=_positive_int,
        metavar="SECONDS",
        help=(
            "per-player time budget in seconds for a timed human-vs-human game; "
            "cannot be combined with --ai"
        ),
    )
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    ai_player = _AI_COLORS.get(args.ai)
    play(secs=args.time, ai_player=ai_player)


if __name__ == "__main__":
    main()
