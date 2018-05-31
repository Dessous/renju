import abc
import numpy
import subprocess
import util
from keras.models import load_model

class Agent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def name(self):
        '''return name of agent'''

    def is_human(self):
        '''false or true'''

    def is_tree(self):
        '''false or true'''

    def get_pos(self, game):
        '''return position for move'''

    def reset(self):
        '''reset for tree'''


class RolloutAgent(Agent):
    def __init__(self, rollout_path, name='Rollout'):
        model = load_model(rollout_path)
        self._name = name
        self.weights = model.layers[1].get_weights()[0]
        self.biases = model.layers[1].get_weights()[1]

    def name(self):
        return self._name

    def is_human(self):
        return False

    def is_tree(self):
        return False

    def get_pos(self, game):
        inp = numpy.zeros((15, 15, 3), dtype=numpy.int8)
        inp[:, :, 0][game.board() == -1] = 1
        inp[:, :, 1][game.board() == 1] = 1
        if game.player() == -1:
            inp[:, :, 2] = 1
        else:
            inp[:, :, 2] = 0
            inp[:, :, [0, 1]] = inp[:, :, [1, 0]]
        inp = inp.flatten()
        probs = (numpy.dot(inp, self.weights) + self.biases).reshape((1, 225))
        probs = util.softmax(probs)
        moves = numpy.argmax(probs.reshape((75, 3)), axis=0)
        for i in range(3):
            moves[i] = moves[i] * 3 + i
        while True:
            pos = numpy.random.choice(moves)
            if not game.is_possible_move((pos // 15, pos % 15)):
                pass
            else:
                break
        return pos // 15, pos % 15

    def reset(self):
        pass


class HumanAgent(Agent):
    def __init__(self, name='Human'):
        self._name = name
        self.pos = None

    def is_human(self):
        return True

    def get_pos(self, game):
        return self.pos

    def name(self):
        return self._name

    def reset(self):
        pass


class RandomAgent(Agent):
    def __init__(self, name='Random'):
        self._name = name

    def name(self):
        return self._name

    def get_pos(self, game):
        num = numpy.random.randint(225)
        pos = num // 15, num % 15
        while not game.is_possible_move(pos):
            num = numpy.random.randint(225)
            pos = num // 15, num % 15
        return pos

    def reset(self):
        pass


class SLAgent(Agent):
    def __init__(self, path, name='SL agent'):
        self._name = name
        self._model = load_model(path)
        self.pos = None

    def name(self):
        return self._name

    def is_human(self):
        return False

    def get_probs(self, game):
        inp = numpy.zeros((15, 15, 3), dtype=numpy.int8)
        inp[:, :, 0][game.board() == -1] = 1
        inp[:, :, 1][game.board() == 1] = 1
        if game.player() == -1:
            inp[:, :, 2] = 1
        else:
            inp[:, :, 2] = 0
            inp[:, :, [0, 1]] = inp[:, :, [1, 0]]
        probs = self._model.predict(numpy.expand_dims(inp, axis=0))
        return probs

    def get_pos(self, game):
        probs = self.get_probs(game)
        probs = probs.reshape(15,15)
        probs[game._board == 1] = 0
        probs[game._board == -1] = 0
        probs = probs.reshape(1, 225) / probs.sum()
        moves = numpy.argmax(probs.reshape((45, 5)), axis=0)
        for i in range(5):
            moves[i] = moves[i] * 5 + i
        while True:
            assert(probs[0, moves].sum() != 0), "bug with probs\n"
            pos = numpy.random.choice(moves, 1,
                                      p=probs[0, moves] / probs[0, moves].sum())
            pos = pos[0]
            if not game.is_possible_move((pos // 15, pos % 15)):
                probs[0, pos] = 0
            else:
                break
        return pos // 15, pos % 15

    def reset(self):
        pass


class BackendAgent(Agent):
    def __init__(self, backend, name='BackendAgent', **kvargs):
        self._name = name
        self._backend = subprocess.Popen(
            backend.split(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            **kvargs
        )

    def name(self):
        return self._name

    def send_game_to_backend(self, game):
        data = game.dumps().encode()
        self._backend.stdin.write(data + b'\n')
        self._backend.stdin.flush()

    def wait_for_backend_move(self):
        data = self._backend.stdout.readline().rstrip()
        return data.decode()

    def get_pos(self, game):
        pass
        # TODO: get position for backend agent

    def reset(self):
        pass