from src.mapper.MnEmbeddingMapper import MnEmbeddingMapper


class OneThresholdEvaluator(MnEmbeddingMapper):
    def __init__(self, threshold: float):
        self.__threshold = threshold

    def map_sentences(self) -> list:
        candidates = []
        for idx1, sentence1 in enumerate(self.article1.sentences):
            for idx2, sentence2 in enumerate(self.article2.sentences):
                if self.similarity_matrix[idx1, idx2] >= self.__threshold:
                    candidates.append((self.similarity_matrix[idx1, idx2].item(), sentence1, sentence2))
        return candidates
