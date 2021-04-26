from time import sleep

from selenium.webdriver.common.keys import Keys

from network import Network


class Player:
    def __init__(self, name, station):
        self.station = station
        self.name = name

    moves = []
    network = Network()

    @staticmethod
    def generate_child(self, genitore1, genitore2):
        pass

    def test(self):
        for move in self.moves:
            move()
            sleep(0.1)


class Player2048(Player):
    def __init__(self, name, station):
        super().__init__(name, station)
        # maybe change the way moves are added, maybe
        self.moves.append(lambda: self.station.game_window().send_keys(Keys.UP))
        self.moves.append(lambda: self.station.game_window().send_keys(Keys.LEFT))
        self.moves.append(lambda: self.station.game_window().send_keys(Keys.DOWN))
        self.moves.append(lambda: self.station.game_window().send_keys(Keys.RIGHT))

    def get_score(self):
        return self.station.get_game_state()["score"]
