import random

import keras.backend
import numpy as np
from keras import Sequential
from keras.layers import Dense, Flatten, InputLayer


class Network:
    model: Sequential

    def __init__(self, nn_params=None):
        if nn_params is None:
            nn_params = {}
        self.accuracy = 0.
        self.nn_params = nn_params
        self.network = {}
        self.model = Sequential()

    def create_random(self):
        for key in self.nn_params:
            self.network[key] = random.choice(self.nn_params[key])

    def create_random_model(self, n_outputs, input_shape):
        # self.create_random()
        minmax = 5
        nb_layers = self.nn_params['nb_layers']
        nb_neurons = self.nn_params['nb_neurons']
        activation = self.nn_params['activation']
        # self.model.add(BatchNormalization(
        #     input_shape=input_shape,
        #     batch_size=1))
        self.model.add(InputLayer(input_shape=input_shape))
        self.model.add(Flatten())
        # self.model.add(Dense(nb_neurons,
        #                      batch_size=1,
        #                      # input_shape=input_shape,
        #                      activation=activation,
        #                      # kernel_initializer=tf.keras.initializers.random_uniform(-minmax, minmax),
        #                      kernel_initializer='random_normal',
        #                      # kernel_initializer='random_uniform',
        #                      # bias_initializer=tf.keras.initializers.random_uniform(-minmax, minmax)
        #                      bias_initializer='random_normal'
        #                      # bias_initializer='random_uniform'
        #                      )
        #                )
        for i in range(nb_layers):
            self.model.add(Dense(nb_neurons,
                                 activation=activation,
                                 # kernel_initializer=tf.keras.initializers.random_uniform(-minmax, minmax),
                                 kernel_initializer='random_normal',
                                 # kernel_initializer='random_uniform',
                                 # bias_initializer=tf.keras.initializers.random_uniform(-minmax, minmax)
                                 bias_initializer='random_normal'
                                 # bias_initializer='random_uniform'
                                 )
                           )

        # self.model.add(Flatten())
        self.model.add(Dense(n_outputs,
                             activation="softmax",
                             # kernel_initializer=tf.keras.initializers.random_uniform(-minmax, minmax),
                             kernel_initializer='random_normal',
                             # kernel_initializer='random_uniform',
                             # bias_initializer=tf.keras.initializers.random_uniform(-minmax, minmax)
                             bias_initializer='random_normal'
                             # bias_initializer='random_uniform'
                             )
                       )

        # self.model.summary()

    def get_next_move_index(self, input_data):
        prediction = self.model(input_data)
        return keras.backend.argmax(prediction[0])

    def update_weights(self, mutation_rate=0.05):
        for layer in self.model.layers:
            weights = layer.get_weights()
            new_weights = []
            for weight in weights:
                # new_weights.append(np.add(weight,
                #                           np.random.normal(
                #                               scale=mutation_rate,
                #                               size=weight.shape)
                #                           )
                #                    )
                for idx, x in np.ndenumerate(weight):
                    if random.random() < 0.2:
                        weight[idx] = x + np.random.choice([-mutation_rate, mutation_rate])

                new_weights.append(weight)

            layer.set_weights(new_weights)
