import json
import random
from time import sleep

import keras.backend as K
import numpy as np
from keras.models import clone_model
from selenium.webdriver.common.keys import Keys

from Game2048Mem import Game2048Mem, Board
from network import Network


class Player:
    network: Network

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
                'nb_neurons': 64,
                'nb_layers': 3,
                'activation': 'sigmoid',
            }
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
    def __init__(self, name, station, mutation_rate=0.1, network=None):
        super().__init__(name, station, mutation_rate, network)
        self.highest_cell = 0
        # maybe change the way moves are added, maybe
        self.moves = []
        self.moves.append(lambda: self.station.game_window().send_keys(Keys.UP))
        self.moves.append(lambda: self.station.game_window().send_keys(Keys.RIGHT))
        self.moves.append(lambda: self.station.game_window().send_keys(Keys.DOWN))
        self.moves.append(lambda: self.station.game_window().send_keys(Keys.LEFT))
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

            matrix = np.log2(matrix)
            matrix = K.constant(np.where(matrix == -np.inf, 0.5, matrix))

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
        max_games = 25
        n_games = 0
        # while n_games < max_games:
        while True:
            next_move = self.network.get_next_move_index(matrix)
            self.do_next_move(next_move)
            new_mat = self.get_matrix()
            if new_mat is not None and np.array_equal(matrix, new_mat):
                # self.mutate()
                # self.station.restart()
                # n_games += 1
                y = np.multiply(1 / 3, np.ones([1, 4]))
                y[0, next_move] = 0
                self.network.model.fit(new_mat, y, epochs=10, steps_per_epoch=10)
                break
            elif self.is_over():
                break
            else:
                matrix = new_mat
            sleep(0.1)

    def reset_best_score(self):
        self.station.driver.execute_script("localStorage.bestScore=0;")

    def generate_child_with(self, genitore2, station, name):
        network1 = self.network
        network2: Network = genitore2.network
        new_network = Network()
        new_network.model = clone_model(network1.model)
        # for i in range(len(network1.model.layers)):
        #     # new_network.model.add(random.choice(list([network1.model.layers[i], network2.model.layers[i]])))
        #     layer1: Layer = network1.model.get_weights()
        #     layer2 = network2.model.get_weights()

        weights1 = network1.model.get_weights()
        weights2 = network2.model.get_weights()
        new_weights = []
        for i, weight1 in enumerate(weights1):
            weight2 = weights2[i]
            new_weight = np.zeros(weight1.shape)
            is_2d = new_weight.ndim > 1
            for x in range(len(new_weight)):
                if is_2d:
                    for y in range(len(new_weight[x])):
                        new_weight[x, y] = random.choice(list([weight1[x, y], weight2[x, y]]))
                else:
                    new_weight[x] = random.choice(list([weight1[x], weight2[x]]))

            new_weights.append(new_weight)

        new_network.model.set_weights(new_weights)
        return Player2048(name, station, network=new_network)


class Player2048Mem:
    def __init__(self, name, mutation_rate=0.1, network=None):
        self.name = name
        self.mutation_rate = mutation_rate
        self.highest_cell = 0
        # maybe change the way moves are added, maybe
        self.game = Game2048Mem(Board())
        self.game.start()
        self.moves = []
        self.moves.append(lambda: self.game.link_keys(0))
        self.moves.append(lambda: self.game.link_keys(1))
        self.moves.append(lambda: self.game.link_keys(2))
        self.moves.append(lambda: self.game.link_keys(3))
        if network is None:
            nn_params = {
                'nb_neurons': 32,
                'nb_layers': 2,
                'activation': 'sigmoid',
            }
            self.network = Network(nn_params)
            self.network.create_random_model(4, [4, 4])
        else:
            self.network = network

    def get_score(self):
        return self.game.game_panel.score

    def get_best_score(self):
        return self.get_score()

    def has_won(self):
        return self.game.won

    def get_matrix(self):
        cells = self.game.game_panel.gridCell
        matrix = np.zeros((4, 4))
        for i, column in enumerate(cells):
            for j, cell in enumerate(column):
                if cell is not None:
                    matrix[i][j] = cell
        matrix = np.log2(matrix)
        matrix = K.constant(np.where(matrix == -np.inf, 0.5, matrix))

        if self.highest_cell < np.amax(matrix):
            self.highest_cell = np.amax(matrix)
        matrix = K.reshape(matrix, [1, 4, 4])
        return matrix

    def is_over(self):
        return self.game.end

    def do_next_move(self, n):
        self.moves[n]()

    def play(self):
        matrix = self.get_matrix()
        max_games = 25
        n_games = 0
        # while n_games < max_games:
        while True:
            next_move = self.network.get_next_move_index(matrix)
            self.do_next_move(next_move)
            new_mat = self.get_matrix()
            if (new_mat is not None and np.array_equal(matrix, new_mat)) or self.is_over():
                break
            else:
                matrix = new_mat
            sleep(0.1)

    def reset_best_score(self):
        self.game.game_panel = Board()
        self.game.start()

    def generate_child_with(self, genitore2, name):
        network1 = self.network
        network2: Network = genitore2.network
        new_network = Network()
        new_network.model = clone_model(network1.model)
        # for i in range(len(network1.model.layers)):
        #     # new_network.model.add(random.choice(list([network1.model.layers[i], network2.model.layers[i]])))
        #     layer1: Layer = network1.model.get_weights()
        #     layer2 = network2.model.get_weights()

        weights1 = network1.model.get_weights()
        weights2 = network2.model.get_weights()
        new_weights = []
        for i, weight1 in enumerate(weights1):
            weight2 = weights2[i]
            new_weight = np.zeros(weight1.shape)
            is_2d = new_weight.ndim > 1
            for x in range(len(new_weight)):
                if is_2d:
                    for y in range(len(new_weight[x])):
                        new_weight[x, y] = random.choice(list([weight1[x, y], weight2[x, y]]))
                else:
                    new_weight[x] = random.choice(list([weight1[x], weight2[x]]))

            new_weights.append(new_weight)

        new_network.model.set_weights(new_weights)
        return Player2048Mem(name, network=new_network)

    def mutate(self):
        self.network.update_weights(self.mutation_rate)
