import json

from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement


class Station:
    driver: WebDriver

    def __init__(self, game):
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        self.driver = webdriver.Chrome(options=options)
        self.game = game
        game.set_driver(self.driver)

    def start(self):
        self.driver.get(self.game.url)

    def restart(self):
        self.game.refresh()

    def shutdown(self):
        self.driver.close()

    def game_window(self) -> WebElement:
        return self.game.game_container()

    def get_game_state(self):
        return json.loads(self.driver.execute_script("return localStorage.gameState;"))

    def get_matrix(self):
        game_state = self.get_game_state()
        cells = game_state["grid"]["cells"]
        matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        for i, column in enumerate(cells):
            for j, cell in enumerate(column):
                if cell is not None:
                    matrix[i][j] = 0
                else:
                    matrix[i][j] = cell["value"]
