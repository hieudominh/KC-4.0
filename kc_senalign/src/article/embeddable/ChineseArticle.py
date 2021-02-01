import re

from src.article.ArticleFactory import CHINESE_LANGID
from src.article.EmbeddableArticle import EmbeddableArticle


class ChineseArticle(EmbeddableArticle):
    sentence_separator = u'[^!?。\.\!\?]+[!?。\.\!\?]?'

    def __init__(self):
        super().__init__(CHINESE_LANGID)

    def __split(self):
        self.sentences += re.findall(self.sentence_separator, self.contents, flags=re.U)
        if '' in self.sentences:
            self.sentences.remove('')
