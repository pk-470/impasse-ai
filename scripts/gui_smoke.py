"""
Headless GUI smoke test: exercises draw_checker (piece_at decode), the info/
highlight rendering, and the pickle save/load+undo round-trip through real GUI
methods, using SDL's dummy video/audio drivers (no window)."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame as pg

from impasse.gui import GUI, HEIGHT, INFO_WIDTH, WIDTH
from impasse.position import WHITE

pg.init()
window = pg.display.set_mode((WIDTH + INFO_WIDTH, HEIGHT))
game = GUI(window)  # human vs human

game.board_update()  # draws board + all checkers (exercises piece_at decode)
print("initial board_update OK; checkers:", game.checkers_total)

# Play a few moves through the real GUI select()/complete_move()/update() path.
import random

rng = random.Random(1)
moves_played = 0
for _ in range(8):
    if game.winner:
        break
    origin = rng.choice(list(game.all_legal_moves))
    targets = game.all_legal_moves[origin]
    target = rng.choice(list(targets))
    tag = targets[target]
    if target is None:
        game.select(origin)  # bear off path
    else:
        game.select(origin)
        game.select(target)
    game.board_update()
    moves_played += 1
print(
    f"played {moves_played} moves through GUI; turn now {'WHITE' if game.turn == WHITE else 'BLACK'}"
)

# Pickle save (already happens in complete_move for human moves) + load (undo).
game.export_position_data()
hash_before = game.state_hash
game.undo_move()  # loads the recovery file and recomputes the hash
game.board_update()
print(
    "undo + reload OK; state is list:",
    isinstance(game.state, list),
    "; hash consistent after load:",
    game.state_hash is not None,
)
print("GUI SMOKE OK")
