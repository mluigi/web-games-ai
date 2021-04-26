import os

from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement


class Game:
    driver: WebDriver

    def set_driver(self, driver):
        self.driver = driver

    def refresh(self):
        pass

    def game_container(self) -> WebElement:
        return self.driver.find_element_by_css_selector("body")


class Game2048(Game):
    url = "file:///{}/2048/index.html".format(os.path.dirname(os.path.realpath(__file__)))

    def refresh(self):
        self.driver.find_element_by_class_name("restart-button").click()
