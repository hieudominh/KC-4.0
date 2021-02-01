from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait

from src.browser.Browser import Browser

CHROME_DRIVER_PATH = 'lib/webdriver/google-chrome/chromedriver'


class Chrome(Browser):
    def __init__(self):
        self.driver = webdriver.Chrome(CHROME_DRIVER_PATH)

    def load_web_page(self, url: str):
        self.driver.get(url)

    def run_and_wait(self, timeout: int, method):
        return WebDriverWait(self.driver, timeout).until(method)

    def close(self):
        self.driver.quit()
