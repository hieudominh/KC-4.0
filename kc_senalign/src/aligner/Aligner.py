from abc import abstractmethod

from src.mapper.Mappable import Mappable


class Aligner:

    def __init__(self):
        self.langid1 = None
        self.article1 = None

        self.langid2 = None
        self.article2 = None

        self.mapper = None

    def set_mapper(self, mapper: Mappable):
        self.mapper = mapper

    def set_article_langid_pair(self, langid1: str, langid2: str):
        self.langid1 = langid1
        self.langid2 = langid2

    def set_article_pair(self, article1: str, article2: str):
        self.article1 = article1
        self.article2 = article2

    @abstractmethod
    def align(self):
        pass

    @abstractmethod
    def stop(self):
        pass
