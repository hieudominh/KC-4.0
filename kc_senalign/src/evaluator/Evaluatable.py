import torch

from src.article.Article import Article


class Evaluable:
    def evaluate(self, article1: Article, article2: Article, similarity_matrix: torch.Tensor) -> list:
        pass
