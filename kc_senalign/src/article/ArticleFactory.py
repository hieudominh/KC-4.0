from src.article.Article import Article


class ArticleFactory:

    def from_text(self, text: str, langid: str) -> Article:
        pass

    def from_sentences(self, sentences: list, langid: str) -> Article:
        pass


CHINESE_LANGID = 'zh'
KHMER_LANGID = 'km'
VIETNAMESE_LANGID = 'vi'