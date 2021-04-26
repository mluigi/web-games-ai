import logging
from time import sleep

from matplotlib import pyplot as plt

from game import Game2048
from player import Player2048
from station import Station

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    filename='log.txt'
)
n_players = 5
stations = []
players = []

# for i in range(n_players):
#     stations.append(Station(Game2048()))
#     stations[i].start()
#     sleep(2)
#     players.append(Player2048("player {}".format(i), stations[i]))


station = Station(Game2048())
station.start()
sleep(2)
player = Player2048("player", station, 0.01)
player.play()
plt.figure(figsize=(9, 3))
plt.bar("player1", player.get_score())
plt.ylabel("score")
plt.show()
station.shutdown()
