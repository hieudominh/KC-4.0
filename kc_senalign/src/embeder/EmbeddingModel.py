from abc import abstractmethod

from src.sentence.Sentence import Sentence


class EmbeddingModel:
    @abstractmethod
    def embed_text(self, sentence: Sentence):
        pass
