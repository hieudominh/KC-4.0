from abc import abstractmethod

from src.article.EmbeddableArticle import EmbeddableArticle
from src.aligner.Aligner import Aligner
from src.article.Article import Article
from src.article.embeddable.EmbeddableArticleFactory import EmbeddableArticleFactory
from src.article.ArticleFactory import VIETNAMESE_LANGID
from src.browser.Chrome import Chrome
from src.embeder.PhoBert import PhoBert
from src.translator.GoogleTranslate import GoogleTranslate


class GoogleTranslatePhoBertAligner(Aligner):

    def __init__(self, device):
        super().__init__()

        self.__device = device
        self._embedder = PhoBert(self.__device)

        self.__browser = Chrome()
        self.__translator = GoogleTranslate(self.__browser)
        self.__translator.set_target_langid(VIETNAMESE_LANGID)
        self.__article_factory = EmbeddableArticleFactory()

        self.similarity_matrix = None

    def set_article_langid_pair(self, langid1: str, langid2: str):
        super().set_article_langid_pair(langid1, langid2)
        if self.langid1 == VIETNAMESE_LANGID and self.langid2 != VIETNAMESE_LANGID:
            self.__translator.set_source_langid(self.langid2)
        elif self.langid1 != VIETNAMESE_LANGID and self.langid2 == VIETNAMESE_LANGID:
            self.__translator.set_source_langid(self.langid1)
        else:
            return
        self.__translator.update()

    def __translate_article(self, article: Article):
        translated_sentences = []
        for sentence in article.sentences:
            translated_sentences.append(self.__translator.translate(sentence))
        # print(translated_sentences)
        return self.__article_factory.from_sentences(translated_sentences, self.__translator.target_langid)

    def align(self):
        # print(self.langid1, '\n\n', self.article1,'\n\n\n', self.langid2, '\n\n', self.article2, '\n\n\n')
        article1 = self.__article_factory.from_text(self.article1, self.langid1)
        article2 = self.__article_factory.from_text(self.article2, self.langid2)
        # print(article1)
        # print(article2)

        # translate article(s)
        translation1 = article1
        translation2 = article2
        if self.langid1 == VIETNAMESE_LANGID and self.langid2 != VIETNAMESE_LANGID:
            translation2 = self.__translate_article(article2)
        elif self.langid1 != VIETNAMESE_LANGID and self.langid2 == VIETNAMESE_LANGID:
            translation1 = self.__translate_article(article1)
        elif self.langid1 != VIETNAMESE_LANGID and self.langid2 != VIETNAMESE_LANGID:
            translation1 = self.__translate_article(article1)
            translation2 = self.__translate_article(article2)

        result = self._process_alignment(article1, article2, translation1, translation2)
        del translation1
        del translation2
        return result

    @abstractmethod
    def _process_alignment(self, article1: Article, article2: Article, translation1: EmbeddableArticle, translation2: EmbeddableArticle):
        pass

    def stop(self):
        self.__browser.close()
