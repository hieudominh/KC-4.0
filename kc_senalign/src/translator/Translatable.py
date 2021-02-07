from abc import abstractmethod


class Translatable:
    def __init__(self):
        self.source_langid = None
        self.target_langid = None

    def set_source_langid(self, langid: str):
        self.source_langid = langid

    def set_target_langid(self, langid: str):
        self.target_langid = langid

    @abstractmethod
    def translate(self, text: str) -> str:
        pass
