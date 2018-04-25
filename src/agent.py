import abc
import numpy
import subprocess
import util
from keras.models import load_model

class Agent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def policy(self, game):
        '''Return probabilty matrix of possible actions'''

    @abc.abstractmethod
    def name(self):
        '''return name of agent'''

class HumanAgent(Agent):
    def __init__(self, name='Human'):
        self._name = name

    def name(self):
        return self._name

    def policy(self, game):
        move = input()
        pos = util.to_pos(move)

        probs = numpy.zeros(game.shape)
        probs[pos] = 1.0

        return probs

class RandomAgent(Agent):
    def __init__(self, name='Random'):
        self._name = name

    def name(self):
        return self._name

    def policy(self, game):
        num = numpy.random.randint(225)
        pos = num // 15, num % 15

        probs = numpy.zeros(game.shape)
        probs[pos] = 1.0

        return probs

class SLAgent(Agent):
    def __init__(self, path, name='SL agent'):
        self._name = name
        self._model = load_model(path)

    def name(self):
        return self._name

    def policy(self, game):
        inp = numpy.zeros((15, 15, 3), dtype=numpy.int8)
        inp[:, :, 0][game.board() == -1] = 1
        inp[:, :, 1][game.board() == 1] = 1
        if game.player() == -1:
            inp[:,:,2] = 1
        else:
            inp[:,:,2] = 0
            inp[:,:,[0, 1]] = inp[:,:,[1, 0]]
        probs = self._model.predict(numpy.expand_dims(inp, axis=0))
        return probs.reshape((game.height, game.width))

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

    def policy(self, game):
        self.send_game_to_backend(game)
        pos = util.to_pos(self.wait_for_backend_move())

        probs = numpy.zeros(game.shape)
        probs[pos] = 1.0

        return probs