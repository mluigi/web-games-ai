import json
from time import sleep

import numpy as np
import tensorflow.keras.backend as K
from selenium.webdriver.common.keys import Keys

from network import Network


class Player:
    def __init__(self, name, station, mutation_rate=0.05):
        self.station = station
        self.name = name
        self.mutation_rate = mutation_rate
        nn_params = {
            'nb_neurons': [64, 128, 256, 512, 768, 1024],
            'nb_layers': [2, 3, 4],
            'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
            'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                          'adadelta', 'adamax', 'nadam'],
        }
        self.network = Network(nn_params)

    moves = []

    @staticmethod
    def generate_child(self, genitore1, genitore2):
        pass

    def test(self):
        for move in self.moves:
            move()
            sleep(0.1)

    def mutate(self):
        self.network.update_weights(self.mutation_rate)


class Player2048(Player):
    def __init__(self, name, station, mutation_rate=0.05):
        super().__init__(name, station, mutation_rate)
        # maybe change the way moves are added, maybe
        self.moves.append(lambda: self.station.game_window().send_keys(Keys.UP))
        self.moves.append(lambda: self.station.game_window().send_keys(Keys.LEFT))
        self.moves.append(lambda: self.station.game_window().send_keys(Keys.DOWN))
        self.moves.append(lambda: self.station.game_window().send_keys(Keys.RIGHT))
        self.network.create_random_model(4, [4, 4])

    def get_score(self):
        return self.get_game_state()["score"]

    def get_game_state(self):
        try:
            return json.loads(self.station.driver.execute_script("return localStorage.gameState;"))
        except Exception as a:
            return None

    def get_matrix(self):
        game_state = self.get_game_state()
        cells = game_state["grid"]["cells"]
        matrix = np.zeros((4, 4))
        for i, column in enumerate(cells):
            for j, cell in enumerate(column):
                if cell is not None:
                    matrix[i][j] = cell["value"]
        matrix = K.constant(matrix)
        K.reshape(matrix, [-1])
        return matrix

    def is_over(self):
        return self.get_game_state() is None

    def do_next_move(self, n):
        self.moves[n]()

    def play(self):
        prev_mat = self.get_matrix()
        while not self.is_over():

            matrix = self.get_matrix()
            if np.array_equal(prev_mat, matrix):
                self.mutate()
                self.station.restart()
            else:
                prev_mat = matrix
            next_move = self.network.get_next_move_index(matrix)

            self.do_next_move(next_move)
            sleep(0.05)
