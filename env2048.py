import json
from time import sleep

import numpy as np
from selenium.webdriver.common.keys import Keys
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from Game2048Mem import Game2048Mem, Board
from game import Game2048
from station import Station


class Env2048(py_environment.PyEnvironment):

    def __init__(self, evaluation_mode: bool = False):
        super().__init__()
        self._evaluation_mode = evaluation_mode
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4, 4, 1), dtype=np.float32, minimum=1, maximum=12, name='observation')
        self._state = np.zeros((4, 4, 1))
        self._episode_ended = False
        self._moves = []
        self.best_score = 0

        if self._evaluation_mode:
            self.station = Station(Game2048())
            self._moves.append(lambda: self.station.game_window().send_keys(Keys.UP))
            self._moves.append(lambda: self.station.game_window().send_keys(Keys.RIGHT))
            self._moves.append(lambda: self.station.game_window().send_keys(Keys.DOWN))
            self._moves.append(lambda: self.station.game_window().send_keys(Keys.LEFT))
        else:
            self.game = Game2048Mem(Board())
            self.game.start()
            self._moves.append(lambda: self.game.link_keys(0))
            self._moves.append(lambda: self.game.link_keys(1))
            self._moves.append(lambda: self.game.link_keys(2))
            self._moves.append(lambda: self.game.link_keys(3))
        self.score = 0

    def observation_spec(self) -> types.NestedArraySpec:
        return self._observation_spec

    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec

    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        if self._episode_ended:
            return self.reset()
        score = self.get_score()
        matrix = self.get_matrix()
        self._moves[action]()
        new_score = self.get_score()
        if new_score > self.best_score:
            self.best_score = new_score
        new_matrix = self.get_matrix()
        self._state = new_matrix if new_matrix is not None else matrix
        reward = 0
        if new_matrix is not None and np.array_equal(matrix, new_matrix):
            reward = -1
        else:
            reward = 1 if new_score - score > 0 else 0

        if self.has_won():
            print("Win")
            self._episode_ended = True
            reward = 2
        elif self.is_over():
            self._episode_ended = True
            reward = -2

        if self._evaluation_mode:
            sleep(0.1)
        if self._episode_ended:
            return ts.termination(self._state.astype('float32'), reward)
        else:
            return ts.transition(self._state.astype('float32'), reward=reward, discount=1.0)

    def get_score(self):
        if self._evaluation_mode:
            state = self.get_game_state()
            if state is None:
                return self.score
            self.score = state["score"]
        else:
            self.score = self.game.game_panel.score
        return self.score

    def _reset(self) -> ts.TimeStep:
        if self._evaluation_mode:
            self.station.restart()
        else:
            self.game.game_panel = Board()
            self.game.end = False
            self.game.start()
        self._state = np.zeros((4, 4, 1), dtype=np.float32)
        self._episode_ended = False
        return ts.restart(self._state)

    def has_won(self):
        if self._evaluation_mode:
            has_won = False
            if self.get_game_state() is not None:
                has_won = self.get_game_state()["won"]
            return has_won
        else:
            return self.game.won

    def get_game_state(self):
        try:
            return json.loads(self.station.driver.execute_script("return localStorage.gameState;"))
        except Exception as e:
            return None

    def is_over(self):
        if self._evaluation_mode:
            return self.get_game_state() is None
        else:
            return self.game.end

    def get_matrix(self):
        matrix = np.zeros((4, 4), dtype=np.float32)
        if self._evaluation_mode:
            game_state = self.get_game_state()
            if game_state is not None:
                cells = game_state["grid"]["cells"]
                for i, column in enumerate(cells):
                    for j, cell in enumerate(column):
                        if cell is not None:
                            matrix[i][j] = cell["value"]

            else:
                return None
        else:
            cells = self.game.game_panel.gridCell
            for i, column in enumerate(cells):
                for j, cell in enumerate(column):
                    if cell is not None:
                        matrix[i][j] = cell

        matrix = np.where(matrix == 0, 1, matrix)
        matrix = np.log2(matrix)
        matrix = np.add(matrix, 1)
        if self._evaluation_mode:
            return matrix.transpose().reshape((4, 4, 1)).astype('float32')
        else:
            return matrix.reshape((4, 4, 1)).astype('float32')
