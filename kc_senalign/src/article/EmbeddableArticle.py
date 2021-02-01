from src.article.Article import Article
from src.embeder.EmbeddingModel import EmbeddingModel
from src.sentence.EmbeddableSentence import EmbeddableSentence


class EmbeddableArticle(Article):

    def __init__(self, langid: str):
        super().__init__(langid)
        self.embedded_sentences = []

    def embed_sentences(self, embedder: EmbeddingModel):
        for sentence in self.sentences:
            embeddable_sentence = EmbeddableSentence(sentence, self.langid)
            embeddable_sentence.embed(embedder)
            self.embedded_sentences.append(embeddable_sentence)
