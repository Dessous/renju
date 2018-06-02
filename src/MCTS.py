import numpy as np
import renju
import agent
from copy import copy, deepcopy
import util
import time


class Node:
    BLACK = -1
    WHITE = 1

    def __init__(self, probs, color=None, parent=None, pos=None):
        self.color = color
        self.parent = parent
        self.pos = pos
        self.children = [None] * 225
        self.Qsa = np.zeros(225)
        self.Nsa = np.zeros(225)
        self.probs = probs
        self.terminal = False
        self.leaf = True
        self.subtree_size = 0

    def is_leaf(self):
        return self.leaf

    def add_child(self, move, probs):
        self.children[move] = Node(probs, -self.color, self, move)
        self.leaf = False

    def is_black(self):
        return self.color == Node.BLACK

    def is_white(self):
        return not self.is_black()

    def is_terminal(self):
        return self.terminal

    def __del__(self):
        for child in self.children:
            if child is not None:
                child.parent = None
                child.__del__()


class MCTS(agent.Agent):
    def __init__(self, agent_path,
                 rollout_path,
                 rollout_depth=10,
                 time=13,
                 name='Tree',
                 linear=False,
                 sim_number=10):
        self.agent = agent.SLAgent(agent_path)
        if linear:
            self.rollout_net = agent.RolloutAgent(rollout_path)
        else:
            self.rollout_net = agent.SLAgent(rollout_path)
        self.rollout_depth = rollout_depth
        self._name = name
        self._time = time
        self.sim_number = sim_number
        self._game = renju.Game()
        self.root = Node(self.agent.get_probs(self._game), Node.BLACK)
        self.run = 0

    def name(self):
        return self._name

    def is_human(self):
        return False

    def is_tree(self):
        return True

    def next_node(self, cur_node, game):
        assert cur_node.probs is not None
        A = cur_node.probs * (1e-9 + np.sqrt(np.sum(cur_node.Nsa))) \
            / (1 + cur_node.Nsa) + cur_node.Qsa
        move = np.argmax(A)
        while not game.is_possible_move((move // 15, move % 15)):
            A[0, move] -= 100
            move = np.argmax(A)
        return move

    def update(self, cur_node, res):
        assert cur_node is not None
        while cur_node.parent is not None:
            cur_node.parent.Nsa[cur_node.pos] += 1
            N = cur_node.parent.Nsa[cur_node.pos]
            Q = cur_node.parent.Qsa[cur_node.pos]
            if cur_node.color == res:
                cur_node.parent.Qsa[cur_node.pos] = ((N - 1) * Q - abs(res)) / N
            else:
                cur_node.parent.Qsa[cur_node.pos] = ((N - 1) * Q + abs(res)) / N
            cur_node = cur_node.parent

    def rollout(self, game):
        counter = 0
        while counter < self.rollout_depth:
            counter += 1
            pos = self.rollout_net.get_pos(game)
            self.run += 1
            game.move(pos)
            if not game:
                if game.result() != renju.Player.NONE:
                    return game.result()
                return 0
        return 0

    def simulation(self):
        game = deepcopy(self._game)
        cur_node = self.root
        while not cur_node.is_leaf():
            move = self.next_node(cur_node, game)
            game.move((move // 15, move % 15))
            #print(util.to_move((move // 15, move % 15)), end=' ')
            if cur_node.children[move] is None:
                cur_node.add_child(move, self.agent.get_probs(game))
                self.run += 1
            cur_node.subtree_size += 1
            #print('check', util.to_move((move // 15, move % 15)),
            #      (move // 15, move % 15))
            cur_node = cur_node.children[move]

        if cur_node.is_terminal():
            self.update(copy(cur_node), -cur_node.color)
            return

        move = self.next_node(cur_node, game)
        game.move((move // 15, move % 15))
        cur_node.add_child(move, self.agent.get_probs(game))
        self.run += 1
        #print(util.to_move((move // 15, move % 15)))
        #print('check', util.to_move((move // 15, move % 15)),
        #     (move // 15, move % 15))
        cur_node.subtree_size += 1
        cur_node = cur_node.children[move]
        if not game:
            cur_node.terminal = True

        if cur_node.is_terminal():
            self.update(copy(cur_node), -cur_node.color)
            return

        res = self.rollout(game)
        self.update(cur_node, res)

    def change_root(self, new_root):
        assert new_root is not None
        new_root.parent = None
        self.root.__del__()
        self.root = new_root
        self.root.pos = None

    def get_pos(self, game):
        assert not self.root.is_terminal()
        start_time = time.clock()
        counter = 0
        self.run = 0
        while (time.clock() - start_time) < self._time:
            self.simulation()
            counter += 1
        #print(self.run)
        #print(self.root.subtree_size)
        max_pos = np.argmax(self.root.Nsa)
        new_root = copy(self.root.children[max_pos])
        self.root.children[max_pos] = None
        self.change_root(new_root)
        self._game.move((max_pos // 15, max_pos % 15))
        return max_pos // 15, max_pos % 15

    def update_tree(self, move):
        self._game.move(move)
        if self.root.children[move[0] * 15 + move[1]] is not None:
            new_root = copy(self.root.children[move[0] * 15 + move[1]])
            self.root.children[move[0] * 15 + move[1]] = None
            self.change_root(new_root)
        else:
            new_root = Node(self.agent.get_probs(self._game), -self.root.color)
            self.change_root(new_root)

    def reset(self):
        self._game = renju.Game()
        self.root = Node(self.agent.get_probs(self._game), Node.BLACK)
