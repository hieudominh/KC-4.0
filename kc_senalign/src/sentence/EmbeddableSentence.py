from src.comparator.Comparable import Comparable
from src.embeder.EmbeddingModel import EmbeddingModel
from src.sentence.Sentence import Sentence


class EmbeddableSentence(Sentence):

    def __init__(self, text: str, language_id: str):
        super().__init__(text, language_id)
        self.embedding = None
        self.embedder = None

    def embed(self, embedder: EmbeddingModel):
        self.embedder = embedder
        self.embedding = embedder.embed_sentence(self)

    def cmp(self, other, comparator: Comparable):
        return comparator.cmp(self.embedding, other.embedding)
