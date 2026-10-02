import pygame as pg

from impasse.gui import GUI, HEIGHT, INFO_WIDTH, SQUARE_SIZE, WIDTH

FPS = 60
SEC = pg.USEREVENT


def get_cell_from_mouse(pos):
    """Return the board cell under a mouse pixel position, or None if off the board."""
    cell = (pos[0] // SQUARE_SIZE, (HEIGHT - pos[1]) // SQUARE_SIZE)
    if 0 <= cell[0] < 8 and 0 <= cell[1] < 8:
        return cell
    return None


def play(secs=None, ai_player=None, dev=False):
    """Open the window and run the game's main event loop until quit."""
    WINDOW = pg.display.set_mode((WIDTH + INFO_WIDTH, HEIGHT))
    pg.display.set_caption("IMPASSE")
    run = True
    clock = pg.time.Clock()
    game = GUI(WINDOW, secs, ai_player, dev=dev)
    pg.time.set_timer(SEC, 1000)

    while run:
        clock.tick(FPS)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False

            if event.type == SEC:
                game.update_time()

            if event.type == pg.MOUSEBUTTONDOWN:
                pos = pg.mouse.get_pos()
                cell = get_cell_from_mouse(pos)
                if cell is not None:
                    game.select(cell)

            if event.type == pg.KEYDOWN:
                if event.key == pg.K_c:
                    game.change_show_cells()

                if event.key == pg.K_z:
                    game.undo_move()

                if event.key == pg.K_n:
                    game.new_game(secs, ai_player)

        game.board_update()

    pg.quit()
