import torch

from src.article.Article import Article
from src.evaluator.Evaluatable import Evaluable


class TopPairEvaluator(Evaluable):
    def __init__(self, pair_number: int):
        self.__pair_number = pair_number

    def evaluate(self, article1: Article, article2: Article, similarity_matrix: torch.Tensor) -> list:
        candidates = []
        pairs = []

        for idx1, sentence1 in enumerate(article1.sentences):
            for idx2, sentence2 in enumerate(article2.sentences):
                pairs.append((similarity_matrix[idx1, idx2].item(), sentence1, sentence2))
            pairs.sort(key=lambda pair: pair[0], reverse=True)

            candidates.extend(pairs[:self.__pair_number])
            pairs.clear()
        return candidates
