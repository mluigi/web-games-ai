# File for testing functions
import random

import numpy as np

from network import Network

nn_params = {
    'nb_neurons': 8,
    'nb_layers': 2,
    'activation': 'sigmoid',
}
network1 = Network(nn_params)
network2 = Network(nn_params)

network1.create_random_model(4, [4, 4])
network2.create_random_model(4, [4, 4])

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

    print("w1==new")
    print(np.equal(new_weight, weight1))
    new_weights.append(new_weight)

for i in range(len(new_weights)):
    print(np.equal(new_weights[i], weights1[i]))
