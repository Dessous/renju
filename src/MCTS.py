import numpy as np
import renju
import agent
from copy import copy, deepcopy
import util


class Node:
    BLACK = -1
    WHITE = 1

    def __init__(self, color=None, parent=None, pos=None):
        self.color = color
        self.parent = parent
        self.pos = pos
        self.children = [None] * 225
        self.Qsa = np.zeros(225)
        self.Nsa = np.zeros(225)
        self.terminal = False
        self.leaf = True
        self.subtree_size = 0

    def is_leaf(self):
        return self.leaf

    def add_child(self, move):
        self.children[move] = Node(-self.color, self, move)
        self.leaf = False

    def is_black(self):
        return self.color == Node.BLACK

    def is_white(self):
        return not self.is_black()

    def is_terminal(self):
        return self.terminal


class MCTS(agent.Agent):
    def __init__(self, agent_path, rollout_path, name='Tree', sim_number=10):
        self.agent = agent.SLAgent(agent_path)
        self.rollout_net = agent.SLAgent(rollout_path)
        self._name = name
        self.sim_number = sim_number
        self.root = Node(Node.BLACK)
        self.game = renju.Game()

    def name(self):
        return self._name

    def is_human(self):
        return False

    def is_tree(self):
        return True

    def next_node(self, cur_node, game):
        probs = self.agent.get_probs(game)
        A = probs / (1 + cur_node.Nsa) + cur_node.Qsa
        move = np.argmax(A)
        while not game.is_possible_move((move // 15, move % 15)):
            A[0, move] -= 100
            move = np.argmax(A)
        return move

    def update(self, cur_node, reward):
        assert cur_node is not None
        while cur_node.parent is not None:
            cur_node.parent.Nsa[cur_node.pos] += 1
            N = cur_node.parent.Nsa[cur_node.pos]
            Q = cur_node.parent.Qsa[cur_node.pos]
            cur_node.parent.Qsa[cur_node.pos] = (N * Q + reward) / (N + 1)
            cur_node = cur_node.parent

    def rollout(self, cur_node, game):
        while True:
            pos = self.rollout_net.get_pos(game)
            game.move(pos)
            if not game:
                if (game.result() == renju.Player.BLACK
                    and cur_node.color == Node.BLACK) \
                    or (game.result() == renju.Player.WHITE
                        and cur_node.color == Node.WHITE):
                    return 1
                elif game.result() != renju.Player.NONE:
                    return -1
                return 0


    def simulation(self):
        game = deepcopy(self.game)
        cur_node = self.root
        while not cur_node.is_leaf():
            move = self.next_node(cur_node, game)
            game.move((move // 15, move % 15))
            if cur_node.children[move] is None:
                cur_node.add_child(move)
            cur_node.subtree_size += 1
            #print('check', util.to_move((move // 15, move % 15)),
            #      (move // 15, move % 15))
            cur_node = cur_node.children[move]

        if cur_node.is_terminal():
            if cur_node.color != self.root.color:
                self.update(copy(cur_node), 2)
            else:
                self.update(copy(cur_node), -2)
            return

        move = self.next_node(cur_node, game)
        game.move((move // 15, move % 15))
        cur_node.add_child(move)
        #print('check', util.to_move((move // 15, move % 15)),
        #     (move // 15, move % 15))
        cur_node.subtree_size += 1
        cur_node = cur_node.children[move]
        if not game:
            cur_node.terminal = True

        if cur_node.is_terminal():
            if cur_node.color != self.root.color:
                self.update(copy(cur_node), 2)
            else:
                self.update(copy(cur_node), -2)
            return

        res = self.rollout(copy(cur_node), game)
        if (res):
            self.update(cur_node, res)

    def change_root(self, new_root):
        self.root = new_root
        self.root.parent = None
        self.root.pos = None

    def get_pos(self, game):
        assert not self.root.is_terminal()
        for i in range(self.sim_number):
            self.simulation()
        max_pos = 0
        max_nsa = 0
        for i in range(225):
            if self.root.children[i] is not None:
                if self.root.Nsa[self.root.children[i].pos] > max_nsa:
                    max_nsa = self.root.Nsa[self.root.children[i].pos]
                    max_pos = self.root.children[i].pos
        print(self.root.subtree_size)
        self.change_root(self.root.children[max_pos])
        self.game.move((max_pos // 15, max_pos % 15))
        return max_pos // 15, max_pos % 15

    def update_tree(self, move):
        if self.root.children[move[0] * 15 + move[1]] is not None:
            self.change_root(self.root.children[move[0] * 15 + move[1]])
        else:
            self.root = Node(-self.root.color)
        self.game.move(move)

    def reset(self):
        self.root = Node(Node.BLACK)
        self.game = renju.Game()