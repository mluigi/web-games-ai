from time import sleep

from matplotlib import pyplot as plt

from game import Game2048
from player import Player2048
from station import Station

station = Station(Game2048())
station.start()
sleep(2)
player = Player2048("test", station)
player.test()
player.test()
print(station.get_game_state())
print(player.get_score())
plt.figure(figsize=(9, 3))
plt.bar("player1", player.get_score())
plt.ylabel("score")
plt.show()
sleep(5)
station.shutdown()
