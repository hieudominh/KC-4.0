from src.article.ArticleFactory import ArticleFactory, VIETNAMESE_LANGID, KHMER_LANGID, CHINESE_LANGID
from src.article.EmbeddableArticle import EmbeddableArticle
from src.article.embeddable.ChineseArticle import ChineseArticle
from src.article.embeddable.KhmerArticle import KhmerArticle
from src.article.embeddable.VietnameseArticle import VietnameseArticle


class EmbeddableArticleFactory(ArticleFactory):

    def from_text(self, text: str, langid: str) -> EmbeddableArticle:
        if text is None or text == '':
            raise ValueError('Please provide the contents of the article: \n')
        article = self.__generate_article(langid)
        article.from_text(text)
        return article

    def __generate_article(self, langid):
        if langid is None or langid == '':
            raise ValueError('Please provide the language id of the following article: \n')
        elif langid == VIETNAMESE_LANGID:
            return VietnameseArticle()
        elif langid == KHMER_LANGID:
            return KhmerArticle()
        elif langid == CHINESE_LANGID:
            return ChineseArticle()
        else:
            raise ValueError('The input language id is not supported yet!')

    def from_sentences(self, sentences: list, langid: str) -> EmbeddableArticle:
        if sentences is None or len(sentences) == 0:
            raise ValueError('Please provide the contents of the article: \n')
        article = self.__generate_article(langid)
        article.from_sentences(sentences)
        return article
