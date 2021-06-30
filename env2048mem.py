import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from Game2048Mem import Game2048Mem, Board


class Env2048Mem(py_environment.PyEnvironment):
    def __init__(self):
        super().__init__()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4, 4), dtype=np.int32, minimum=0, maximum=2048, name='observation')
        self._state = np.zeros((4, 4), dtype=np.int32)
        self._episode_ended = False
        self.game = Game2048Mem(Board())
        self.game.start()
        self.moves = []
        self.moves.append(lambda: self.game.link_keys(0))
        self.moves.append(lambda: self.game.link_keys(1))
        self.moves.append(lambda: self.game.link_keys(2))
        self.moves.append(lambda: self.game.link_keys(3))
        self.prev_score = 0

    def observation_spec(self) -> types.NestedArraySpec:
        return self._observation_spec

    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec

    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        if self._episode_ended:
            return self.reset()
        score = self.get_score()
        matrix = self.get_matrix()
        self.moves[action]()
        new_score = self.get_score()
        new_matrix = self.get_matrix()
        self._state = new_matrix if new_matrix is not None else matrix
        reward = 0
        if new_matrix is not None and np.array_equal(matrix, new_matrix):
            reward = -10
        elif self.is_over():
            self._episode_ended = True
            reward = -50
        elif self.has_won():
            print("Win")
            self._episode_ended = True
            reward = 1000
        else:
            reward = new_score - score
        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward=reward, discount=1.0)

    def _reset(self) -> ts.TimeStep:
        self.game.game_panel = Board()
        self.game.start()
        self._state = np.zeros((4, 4), dtype=np.int32)
        self._episode_ended = False
        return ts.restart(self._state)

    def get_score(self):
        return self.game.game_panel.score

    def get_best_score(self):
        return self.get_score()

    def has_won(self):
        return self.game.won

    def get_matrix(self):
        cells = self.game.game_panel.gridCell
        matrix = np.zeros((4, 4), dtype=np.int32)
        for i, column in enumerate(cells):
            for j, cell in enumerate(column):
                if cell is not None:
                    matrix[i][j] = cell
        return matrix

    def is_over(self):
        return self.game.end
