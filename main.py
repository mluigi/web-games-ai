import random
from concurrent.futures import ThreadPoolExecutor

from player import Player2048Mem

# logging.basicConfig(
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     datefmt='%m/%d/%Y %I:%M:%S %p',
#     level=logging.DEBUG,
#     filename='log.txt'
# )


n_players = 128
stations = []
players = []
n_pops = 0
# with ThreadPoolExecutor(max_workers=4) as executor:
#     def init():
#         global n_pops
#         stations.append(Station(Game2048()))
#         players.append(Player2048(f"player {n_pops}", stations[i]))
#         n_pops += 1
#
#     for i in range(n_players):
#         executor.submit(init)

for i in range(n_players):
    players.append(Player2048Mem(f"player {n_pops}"))
    n_pops += 1

has_somebody_won = False
generation = 0
while not has_somebody_won:
    with ThreadPoolExecutor(max_workers=n_players) as executor:
        for player in players:
            executor.submit(player.play)

    sorted_players = [player for player in sorted(players, key=lambda x: x.get_best_score(), reverse=True)]
    # sorted_players = [player for player in sorted(players, key=lambda x: x.highest_cell, reverse=True)]

    print(f"generation {generation} top 10")
    print("\t\tScore\tHighest cell")
    for player in sorted_players[:10]:
        print(f"{player.name}\t{player.get_best_score()}\t{player.highest_cell}")

    winners = [player for player in players if player.has_won()]

    if len(winners) > 0:
        has_somebody_won = True
        print("Winners:")
        for winner in winners:
            print(f"{winner.name}: {winner.get_best_score()}\thighest cell: {winner.highest_cell}")
    else:
        retained_players = sorted_players[:int(n_players / 2)]
        removed_players = sorted_players[int(n_players / 2):]
        # available_stations = list(map(lambda x: x.station, removed_players))
        players = retained_players
        while len(players) < n_players:
            genitore1 = random.randint(0, len(players) - 1)
            genitore2 = random.randint(0, len(players) - 1)
            if genitore1 != genitore2:
                genitore1 = players[genitore1]
                genitore2 = players[genitore2]
                new_player = genitore1.generate_child_with(genitore2,
                                                           # available_stations.pop(),
                                                           f"player {n_pops}")
                if random.random() < 0.33:
                    new_player.mutate()
                players.append(new_player)
                n_pops += 1
        for player in players:
            player.reset_best_score()
        #            player.station.restart()

        generation += 1

for station in stations:
    station.shutdown()

# network = Network({
#     'nb_neurons': [64, 128, 256, 512, 768, 1024],
#     'nb_layers': [2, 3, 4, 5, 6],
#     'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
# })
# mat = tensorflow.keras.backend.random_uniform((1, 4, 4), maxval=2048)
# network.create_random_model(4, (4, 4))
# print("mat")
# print(mat)
# print("predict")
# print(network.model(mat))
