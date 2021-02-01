from src.article.ArticleFactory import VIETNAMESE_LANGID
from src.article.EmbeddableArticle import EmbeddableArticle


class VietnameseArticle(EmbeddableArticle):
    sentence_separator = r"(?<!\w[\.]\w[\.])(?<![A-Z][a-z][\.])(?<=[\.]|\?|\!)\s"

    def __init__(self):
        super().__init__(VIETNAMESE_LANGID)
