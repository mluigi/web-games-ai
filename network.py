import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


class Network:

    def __init__(self, nn_params):
        self.accuracy = 0.
        self.nn_params = nn_params
        self.network = {}
        self.model = Sequential()

    def create_random(self):
        """Create a random network."""
        for key in self.nn_params:
            self.network[key] = random.choice(self.nn_params[key])

    def create_random_model(self, n_outputs, input_shape):
        self.create_random()

        nb_layers = self.network['nb_layers']
        nb_neurons = self.network['nb_neurons']
        activation = self.network['activation']
        self.model.add(Dense(nb_neurons,
                             input_shape=input_shape,
                             activation=activation,
                             kernel_initializer=tf.keras.initializers.random_uniform(-1, 1),
                             bias_initializer=tf.keras.initializers.random_uniform(-1, 1)))
        # self.model.add(Flatten())
        for i in range(nb_layers - 1):
            self.model.add(Dense(nb_neurons,
                                 activation=activation,
                                 kernel_initializer=tf.keras.initializers.random_uniform(-1, 1),
                                 bias_initializer=tf.keras.initializers.random_uniform(-1, 1)))

        self.model.add(Dense(n_outputs,
                             activation="softmax",
                             kernel_initializer=tf.keras.initializers.random_uniform(-1, 1),
                             bias_initializer=tf.keras.initializers.random_uniform(-1, 1)))
        # self.model.compile(optimizer=optimizer)

        self.model.summary()

    def get_next_move_index(self, input_data):
        prediction = self.model(input_data)
        return tf.math.argmax(prediction[0])

    def update_weights(self, mutation_rate=0.05):
        for layer in self.model.layers:
            weights = layer.get_weights()
            new_weights = []
            for weight in weights:
                new_weights.append(np.add(weight,
                                          np.random.uniform(
                                              -mutation_rate,
                                              mutation_rate,
                                              size=weight.shape)
                                          )
                                   )
            layer.set_weights(new_weights)
