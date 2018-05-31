import enum
import numpy
import util
import traceback
import sys
import itertools
import logging
import concurrent


class Player(enum.IntEnum):
    NONE = 0
    BLACK = -1
    WHITE = 1

    def another(self):
        return Player(-self)

    def __repr__(self):
        if self == Player.BLACK:
            return 'black'
        elif self == Player.WHITE:
            return 'white'
        else:
            return 'none'

    def __str__(self):
        return self.__repr__()


class Game:
    width, height = 15, 15
    shape = (width, height)
    line_length = 5
    max_game_length = 80

    def __init__(self):
        self.turn_number = 0
        self._result = Player.NONE
        self._player = Player.BLACK
        self._board = numpy.full(self.shape, Player.NONE, dtype=numpy.int8)
        self._positions = list()

    def __bool__(self):
        return self.result() == Player.NONE and \
               self.turn_number < self.max_game_length

    def move_n(self):
        return len(self._positions)

    def player(self):
        return self._player

    def result(self):
        return self._result

    def board(self):
        return self._board

    def positions(self, player=Player.NONE):
        if not player:
            return self._positions

        begin = 0 if player == Player.BLACK else 1
        return self._positions[begin::2]

    def dumps(self):
        return ' '.join(map(util.to_move, self._positions))

    @staticmethod
    def loads(dump):
        game = Game()
        for pos in map(util.to_pos, dump.split()):
            game.move(pos)
        return game

    def is_possible_move(self, pos):
        return 0 <= pos[0] < self.height \
               and 0 <= pos[1] < self.width \
               and not self._board[pos]

    def move(self, pos):
        assert self.is_possible_move(pos), 'impossible pos: {pos}'.format(
            pos=pos)

        #self._positions.append(pos)
        self._board[pos] = self._player

        if not self._result and util.check(self._board, pos):
            self._result = self._player
            return

        self.turn_number += 1
        self._player = self._player.another()

    def update(self, probs):
        pos = 0
        while True:
            num = numpy.argmax(probs)
            pos = num // 15, num % 15
            if not self.is_possible_move(pos):
                probs[pos] -= 1
            else:
                break
        self.move(pos)
        self.print_field()

    def print_field(self):
        board = """
           a   b   c   d   e   f   g   h   j   k   l   m   n   o   p
         =============================================================
       1 | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c |
         |---+---+---+---+---+---+---+---+---+---+---+---+---+---+---|
       2 | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c |
         |---+---+---+---+---+---+---+---+---+---+---+---+---+---+---|
       3 | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c |
         |---+---+---+---+---+---+---+---+---+---+---+---+---+---+---|
       4 | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c |
         |---+---+---+---+---+---+---+---+---+---+---+---+---+---+---|
       5 | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c |
         |---+---+---+---+---+---+---+---+---+---+---+---+---+---+---|
       6 | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c |
         |---+---+---+---+---+---+---+---+---+---+---+---+---+---+---|
       7 | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c |
         |---+---+---+---+---+---+---+---+---+---+---+---+---+---+---|
       8 | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c |
         |---+---+---+---+---+---+---+---+---+---+---+---+---+---+---|
       9 | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c |
         |---+---+---+---+---+---+---+---+---+---+---+---+---+---+---|
      10 | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c |
         |---+---+---+---+---+---+---+---+---+---+---+---+---+---+---|
      11 | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c |
         |---+---+---+---+---+---+---+---+---+---+---+---+---+---+---|
      12 | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c |
         |---+---+---+---+---+---+---+---+---+---+---+---+---+---+---|
      13 | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c |
         |---+---+---+---+---+---+---+---+---+---+---+---+---+---+---|
      14 | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c |
         |---+---+---+---+---+---+---+---+---+---+---+---+---+---+---|
      15 | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c | %c |
         =============================================================
        """
        to_print = self._board.copy()
        to_print *= 79
        to_print[to_print == -79] = 88
        to_print[to_print == 0] = 45
        print(board % tuple(to_print.flatten()))


def loop(game, black, white, timeout=None):
    #yield game, numpy.zeros(game.shape)
    for agent in itertools.cycle([black, white]):
        if not game:
            break

        probs = agent.policy(game)
        yield game, probs

def run_test(black, white, timeout=None):
    game = Game()
    try:
        for game, probs in loop(game, black, white, timeout):
            game.update(probs)

    except:
        _, e, tb = sys.exc_info()
        print(e)
        traceback.print_tb(tb)
        return game.player().another()

    return game.result()

def run(black, white, max_move_n=60, timeout=10):
    game = Game()

    try:
        for game, _ in loop(game, black, white, timeout):
            logging.debug(game.dumps() + '\n' + str(game.board()))
            if game.move_n() >= max_move_n:
                break

    except:
        logging.error('Error!', exc_info=True, stack_info=True)
        return game.player().another(), game.dumps()

    return game.result(), game.dumps()