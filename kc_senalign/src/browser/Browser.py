from abc import abstractmethod


class Browser:

    @abstractmethod
    def load_web_page(self, url: str):
        pass

    @abstractmethod
    def run_and_wait(self, timeout: int, method):
        pass

    @abstractmethod
    def close(self):
        pass
