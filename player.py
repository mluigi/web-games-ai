import json
import random
from time import sleep

import numpy as np
import tensorflow.keras.backend as K
from selenium.webdriver.common.keys import Keys

from network import Network


class Player:
    def __init__(self, name, station, mutation_rate=0.1, network=None):
        self.station = station
        self.name = name
        self.mutation_rate = mutation_rate
        # nn_params = {
        #     'nb_neurons': [64, 128, 256, 512, 768, 1024],
        #     'nb_layers': [2, 3, 4, 5, 6],
        #     'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        # }
        if network is None:
            nn_params = {
                'nb_neurons': 32,
                'nb_layers': 2,
                'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
            }
            nn_params['activation'] = random.choice(nn_params['activation'])
            self.network = Network(nn_params)
        else:
            self.network = network

    def generate_child_with(self, genitore2, station, name):
        pass

    def play(self):
        pass

    def mutate(self):
        self.network.update_weights(self.mutation_rate)


class Player2048(Player):
    def __init__(self, name, station, mutation_rate=0.05, network=None):
        super().__init__(name, station, mutation_rate, network)
        self.highest_cell = 0
        # maybe change the way moves are added, maybe
        self.moves = []
        self.moves.append(lambda: self.station.game_window().send_keys(Keys.UP))
        self.moves.append(lambda: self.station.game_window().send_keys(Keys.LEFT))
        self.moves.append(lambda: self.station.game_window().send_keys(Keys.DOWN))
        self.moves.append(lambda: self.station.game_window().send_keys(Keys.RIGHT))
        if network is None:
            self.network.create_random_model(4, [4, 4])

    def get_score(self):
        return self.get_game_state()["score"]

    def get_best_score(self):
        best_score = self.station.driver.execute_script("return localStorage.bestScore;")
        if best_score is None:
            best_score = "0"
        return int(best_score)

    def has_won(self):
        has_won = False
        if self.get_game_state() is not None:
            has_won = self.get_game_state()["won"]
        return has_won

    def get_game_state(self):
        try:
            return json.loads(self.station.driver.execute_script("return localStorage.gameState;"))
        except Exception as a:
            return None

    def get_matrix(self):
        game_state = self.get_game_state()
        if game_state is not None:
            cells = game_state["grid"]["cells"]
            matrix = np.zeros((4, 4))
            for i, column in enumerate(cells):
                for j, cell in enumerate(column):
                    if cell is not None:
                        matrix[i][j] = cell["value"]
            matrix = K.constant(matrix)
            if self.highest_cell < np.amax(matrix):
                self.highest_cell = np.amax(matrix)
            matrix = K.reshape(matrix, [1, 4, 4])
            return matrix
        else:
            return None

    def is_over(self):
        return self.get_game_state() is None

    def do_next_move(self, n):
        self.moves[n]()

    def play(self):
        matrix = self.get_matrix()
        max_games = 100
        n_games = 0
        while n_games < max_games:
            next_move = self.network.get_next_move_index(matrix)
            self.do_next_move(next_move)
            new_mat = self.get_matrix()
            if new_mat is not None and np.array_equal(matrix, new_mat):
                self.mutate()
                n_games += 1
            elif self.is_over():
                # n_games += 1
                # self.station.restart()
                break
            else:
                matrix = new_mat
            sleep(0.1)

    def reset_best_score(self):
        self.station.driver.execute_script("localStorage.bestScore=0;")

    def generate_child_with(self, genitore2, station, name):
        network1 = self.network
        network2 = genitore2.network
        new_network = Network()
        for i in range(len(network1.model.layers)):
            new_network.model.add(random.choice(list([network1.model.layers[i], network2.model.layers[i]])))

        return Player2048(name, station, network=new_network)
