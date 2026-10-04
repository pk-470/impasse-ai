# Impasse Search Engine

This is a search engine for the game [Impasse](https://www.marksteeregames.com/Impasse_rules.pdf)
by Mark Steere. It uses alpha-beta search with move ordering, iterative deepening
and a transposition table, wrapped in a pygame GUI.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Installation

Clone the repository and let uv create the virtual environment and install the
package (and its dependencies, pygame and platformdirs):

```bash
uv sync
```

### Optional: a stronger AI

Compiling the search engine to C extensions makes it about four times faster,
letting it search one to two plies deeper in the same thinking time. Needs a C
compiler (MSVC on Windows, gcc or clang elsewhere).

```bash
uv run python scripts/build_native.py           # compile
uv run python scripts/build_native.py --check   # show what is in use
uv run python scripts/build_native.py --clean   # go back to pure Python
```

## Playing

Launch a game with `uv run impasse`:

```bash
uv run impasse                 # human vs human, untimed
uv run impasse --ai black      # you play White against the AI
uv run impasse --ai white      # you play Black against the AI (AI moves first)
uv run impasse --time 600      # timed human-vs-human game, 10 minutes each side
uv run impasse --ai black --dev  # print per-move AI search diagnostics to the console
```

`--ai` and `--time` cannot be combined — timed games are human-vs-human only.
`--dev` prints per-move search stats (transposition-table visits/hits/cutoffs, eval-cache
hits, and nodes searched per second) and is most useful together with `--ai`.
Run `uv run impasse --help` for the full list of options. With the environment
active you can equivalently launch the game with `python -m impasse`.

### In-game controls

| Key | Action                                  |
| --- | --------------------------------------- |
| `c` | Show or hide the cell names             |
| `z` | Undo the last move (one move only)      |
| `n` | Start a new game with the same settings |

Before each human move the position is written to a file in your user data
directory (e.g. `~/.local/share/impasse/` on Linux), which is what `z` reads to
undo. It lives outside the package, so no write access to the install location is
needed. Nothing reloads it at startup — it is an undo buffer, not a saved game.

## Project layout

```
src/impasse/
├── __main__.py     # argparse CLI entry point (`impasse` / `python -m impasse`)
├── play.py         # pygame event loop
├── position.py     # board state, move generation, evaluation, Zobrist hashing
├── gui.py          # pygame GUI wrapper
└── ai.py           # alpha-beta search engine

scripts/
├── bench.py        # throughput + cProfile
├── build_native.py # compile position.py and ai.py with mypyc
├── selfplay.py     # full-game integrity check
└── gui_smoke.py    # headless GUI check
```
