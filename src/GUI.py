from tkinter import *
import util
import agent
import renju
import time
import math
import itertools


class Board_Canvas(Canvas):
    def __init__(self, master=None, height=550, width=480):
        Canvas.__init__(self, master, height=height, width=width)

        # initializations
        self.draw_gameBoard()

    '''
        Plot the game board
    '''

    def draw_gameBoard(self):
        self.create_rectangle(30, 30, 15 * 30, 15 * 30, fill="#E9D66B")
        # 15 horizontal lines
        for i in range(15):
            start_pixel_x = (i + 1) * 30
            start_pixel_y = (0 + 1) * 30
            end_pixel_x = (i + 1) * 30
            end_pixel_y = (14 + 1) * 30
            self.create_line(start_pixel_x, start_pixel_y, end_pixel_x,
                             end_pixel_y)

        # 15 vertical lines
        for j in range(15):
            start_pixel_x = (0 + 1) * 30
            start_pixel_y = (j + 1) * 30
            end_pixel_x = (14 + 1) * 30
            end_pixel_y = (j + 1) * 30
            self.create_line(start_pixel_x, start_pixel_y, end_pixel_x,
                             end_pixel_y)

        # place a "star" to particular intersections
        self.draw_star(3, 3)
        self.draw_star(11, 3)
        self.draw_star(7, 7)
        self.draw_star(3, 11)
        self.draw_star(11, 11)

        # coordinates
        self.draw_coords()

    def draw_star(self, row, col):
        start_pixel_x = (row + 1) * 30 - 2
        start_pixel_y = (col + 1) * 30 - 2
        end_pixel_x = (row + 1) * 30 + 2
        end_pixel_y = (col + 1) * 30 + 2

        self.create_oval(start_pixel_x, start_pixel_y, end_pixel_x,
                         end_pixel_y, fill='black')

    def draw_stone(self, row, col, turn):
        start_pixel_x = (row + 1) * 30 - 10
        start_pixel_y = (col + 1) * 30 - 10
        end_pixel_x = (row + 1) * 30 + 10
        end_pixel_y = (col + 1) * 30 + 10

        if turn % 2:
            self.create_oval(start_pixel_x, start_pixel_y, end_pixel_x,
                             end_pixel_y, fill='black')
            self.create_text(start_pixel_x + 10, start_pixel_y + 10,
                             text=str(turn), fill='white')
        else:
            self.create_oval(start_pixel_x, start_pixel_y, end_pixel_x,
                             end_pixel_y, fill='white')
            self.create_text(start_pixel_x + 10, start_pixel_y + 10,
                             text=str(turn), fill='black')

    def draw_coords(self):
        for i, letter in enumerate(util.POS_TO_LETTER):
            self.create_text((i + 1) * 30, 10, text=letter)
        for i in range(1, 16):
            self.create_text(10, i * 30, text=str(i))


class GameUI(Frame):
    def __init__(self, black, white, master=None, delay=0):
        super().__init__(master)
        self.game = renju.Game()
        self._black = black
        self._white = white
        self._player = self._black
        self._turn_number = 1
        self._delay = delay
        self.event_id = 0
        self.board_canvas = Board_Canvas(master)
        self.board_canvas.pack()
        self.game_loop()

    def reset_game(self, event):
        print("Restart!")
        if event.char == 'r' or event.char == 's':
            if event.char == 's':
                self._black, self._white = self._white, self._black
            self._player = self._black
            self._turn_number = 1
            self.game = renju.Game()
            self.board_canvas.delete('all')
            self.board_canvas.draw_gameBoard()
            self.master.unbind('<KeyPress>')
            self.game_loop()

    def next_player(self):
        if self._player is self._black:
            self._player = self._white
        else:
            self._player = self._black

    def read_move(self, event):
        'Process human move on mouse click'
        self.board_canvas.unbind('<Button-1>')
        for i in range(15):
            for j in range(15):
                pixel_x = (i + 1) * 30
                pixel_y = (j + 1) * 30
                square_x = (event.x - pixel_x)**2
                square_y = (event.y - pixel_y)**2
                distance = math.sqrt(square_x + square_y)

                if distance < 15 and self.game.is_possible_move((i, j)):
                    self._player.pos = (i, j)
                    return

    def game_loop(self):
        #print("game start!")
        pos = self._player.get_pos(self.game)
        if pos:
            assert self.game.is_possible_move(pos)
            if self.game:
                if not self._player.is_human():
                    time.sleep(self._delay)
                self.game.move(pos)
                self.board_canvas.draw_stone(*pos, self._turn_number)
                self._turn_number += 1

            if self._player.is_human():
                self._player.pos = None

            if not self.game:
                if self.game.result() == renju.Player.WHITE:
                    winner = self._white.name() + ' [white]'
                else:
                    winner = self._black.name() + ' [black]'
                self.board_canvas.create_text(8 * 30, 16 * 30, text=winner)
                self.after_cancel(self.event_id)
                self.board_canvas.create_text(8 * 30, 16 * 30 + 18,
                                              text='Press (R) for restart')
                self.board_canvas.create_text(8 * 30, 16 * 30 + 30,
                                              text='Press (S) for swap sides')
                self.master.bind('<KeyPress>', self.reset_game)

            self.next_player()

        elif self._player.is_human():
            self.board_canvas.bind('<Button-1>', self.read_move)

        self.event_id = self.after(100, self.game_loop)




def run_gui(black, white, delay=1):
    root = Tk()
    app = GameUI(master=root, black=black, white=white, delay=delay)
    app.master.title("Gomoku")
    root.mainloop()