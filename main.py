import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor

import tensorflow.keras as keras

from network import Network
from player import Player2048Mem


def delete_all_files_in(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def move_all_files(src, dst):
    file_names = os.listdir(src)
    for file_name in file_names:
        shutil.move(os.path.join(src, file_name), dst)


def main():
    n_players = 200
    stations = []
    players = []
    n_pops = 0
    generation = 0

    if not os.path.exists("chkpoints"):
        os.mkdir("chkpoints")
        os.makedirs("chkpoints/prev")
        os.makedirs("chkpoints/lts")
        for i in range(n_players):
            players.append(Player2048Mem(f"player {n_pops}"))
        n_pops += 1
    elif os.path.exists("chkpoints/lts") and len(os.listdir("chkpoints/lts")) > 0:
        print("sasa")
        max_n_players = 0
        for filename in os.listdir("chkpoints/lts"):
            split_filename = filename.split("-")
            gen = split_filename[0]
            player_name = split_filename[1].split(".")[0]
            player_n = int(player_name.split(" ")[1])
            if max_n_players < player_n:
                max_n_players = player_n
            generation = int(gen)
            network = Network()
            network.model = keras.models.load_model(f"chkpoints/lts/{filename}")
            players.append(Player2048Mem(player_name, network=network))
        print("loaded saved players")
    else:
        for i in range(n_players):
            players.append(Player2048Mem(f"player {n_pops}"))
        n_pops += 1

    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     def init():
    #         global n_pops
    #         stations.append(Station(Game2048()))
    #         players.append(Player2048(f"player {n_pops}", stations[i]))
    #         n_pops += 1
    #
    #     for i in range(n_players):
    #         executor.submit(init)

    has_somebody_won = False
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
                winner.network.model.save(winner.name)
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

            generation += 1
            with ThreadPoolExecutor(max_workers=50) as executor:
                def reset(player):
                    player.reset_best_score()
                    if generation % 50 == 0:
                        delete_all_files_in("chkpoints/prev")
                        move_all_files("chkpoints/lts", "chkpoints/prev")
                        player.network.model.save(f"chkpoints/lts/{generation}-{player.name}.keras")

                #   player.station.restart()

                for player in players:
                    executor.submit(reset, player)

    for station in stations:
        station.shutdown()


if __name__ == '__main__':
    main()
