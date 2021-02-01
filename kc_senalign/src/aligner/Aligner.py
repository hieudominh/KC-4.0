from src.evaluator.Evaluatable import Evaluable


class Aligner:

    def __init__(self):
        self.langid1 = None
        self.article1 = None

        self.langid2 = None
        self.article2 = None

        self.evaluator = None

    def set_evaluator(self, evaluator: Evaluable):
        self.evaluator = evaluator

    def set_article_langid_pair(self, langid1: str, langid2: str):
        self.langid1 = langid1
        self.langid2 = langid2

    def set_article_pair(self, article1: str, article2: str):
        self.article1 = article1
        self.article2 = article2

    def align(self):
        pass

    def stop(self):
        pass
