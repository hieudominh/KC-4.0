from abc import abstractmethod

import torch

from src.article.Article import Article
from src.mapper.Mappable import Mappable


class MnEmbeddingMapper(Mappable):

    def __init__(self):
        self.article1 = None
        self.article2 = None
        self.similarity_matrix = None

    def set_params(self, article1: Article, article2: Article, similarity_matrix: torch.Tensor):
        self.article1 = article1
        self.article2 = article2
        self.similarity_matrix = similarity_matrix

    @abstractmethod
    def map_sentences(self) -> list:
        pass
