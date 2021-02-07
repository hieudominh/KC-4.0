from abc import abstractmethod

from src.comparator.Comparable import Comparable


class Sentence:
    def __init__(self, text: str, langid: str):
        self.text = text
        self.langid = langid

    def __str__(self):
        return self.text

    @abstractmethod
    def cmp(self, other, comparator: Comparable):
        pass
