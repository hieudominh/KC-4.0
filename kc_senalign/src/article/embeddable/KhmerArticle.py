from src.article.ArticleFactory import KHMER_LANGID
from src.article.EmbeddableArticle import EmbeddableArticle


class KhmerArticle(EmbeddableArticle):
    sentence_separator = r"(?<!\w[\.៕។]\w[\.៕។])(?<![A-Z][a-z][\.៕។])(?<=[\.៕។]|\?|\!)\s"

    def __init__(self):
        super().__init__(KHMER_LANGID)
