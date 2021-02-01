import torch

from src.comparator.Comparable import Comparable


class CosineSimilarityComparator(Comparable):
    def __init__(self):
        self.__cosine = torch.nn.CosineSimilarity(dim=0)

    def cmp(self, element1, element2):
        return self.__cosine(element1, element2)

