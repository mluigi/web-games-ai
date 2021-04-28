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
        self.driver.get(self.game.url)

    def restart(self):
        self.game.refresh()

    def shutdown(self):
        self.driver.close()

    def game_window(self) -> WebElement:
        return self.game.game_container()
