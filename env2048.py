import json
from time import sleep

import numpy as np
from selenium.webdriver.common.keys import Keys
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from game import Game2048
from station import Station


class Env2048(py_environment.PyEnvironment):
    def __init__(self):
        super().__init__()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4, 4), dtype=np.int32, minimum=0, maximum=2048, name='observation')
        self._state = np.zeros((4, 4), dtype=np.int32)
        self._episode_ended = False
        self.moves = []
        self.moves.append(lambda: self.station.game_window().send_keys(Keys.UP))
        self.moves.append(lambda: self.station.game_window().send_keys(Keys.RIGHT))
        self.moves.append(lambda: self.station.game_window().send_keys(Keys.DOWN))
        self.moves.append(lambda: self.station.game_window().send_keys(Keys.LEFT))
        self.station = Station(Game2048())
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
            reward = -50
        elif self.is_over():
            self._episode_ended = True
            reward = -200
        elif self.has_won():
            self._episode_ended = True
            reward = 5000
        else:
            reward = new_score - score
        sleep(0.1)
        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward=reward, discount=1.0)

    def get_score(self):
        state = self.get_game_state()
        if state is None:
            return self.prev_score
        self.prev_score = state["score"]
        return self.prev_score

    def _reset(self) -> ts.TimeStep:
        self.station.restart()
        self._state = np.zeros((4, 4), dtype=np.int32)
        self._episode_ended = False
        return ts.restart(self._state)

    def has_won(self):
        has_won = False
        if self.get_game_state() is not None:
            has_won = self.get_game_state()["won"]
        return has_won

    def get_game_state(self):
        try:
            return json.loads(self.station.driver.execute_script("return localStorage.gameState;"))
        except Exception as e:
            return None

    def is_over(self):
        return self.get_game_state() is None

    def get_matrix(self):
        game_state = self.get_game_state()
        if game_state is not None:
            cells = game_state["grid"]["cells"]
            matrix = np.zeros((4, 4), dtype=np.int32)
            for i, column in enumerate(cells):
                for j, cell in enumerate(column):
                    if cell is not None:
                        matrix[i][j] = cell["value"]
            return matrix
        else:
            return None
