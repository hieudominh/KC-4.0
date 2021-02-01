import torch

from src.article.Article import Article
from src.evaluator.Evaluatable import Evaluable


class OneThresholdEvaluator(Evaluable):
    def __init__(self, threshold: float):
        self.__threshold = threshold

    def evaluate(self, article1: Article, article2: Article, similarity_matrix: torch.Tensor) -> list:
        candidates = []
        for idx1, sentence1 in enumerate(article1.sentences):
            for idx2, sentence2 in enumerate(article2.sentences):
                if similarity_matrix[idx1, idx2] >= self.__threshold:
                    candidates.append((similarity_matrix[idx1, idx2].item(), sentence1, sentence2))
        return candidates
