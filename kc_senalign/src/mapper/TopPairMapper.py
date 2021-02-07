from src.mapper.MnEmbeddingMapper import MnEmbeddingMapper


class TopPairEvaluator(MnEmbeddingMapper):
    def __init__(self, pair_number: int):
        self.__pair_number = pair_number

    def map_sentences(self) -> list:
        candidates = []
        pairs = []

        for idx1, sentence1 in enumerate(self.article1.sentences):
            for idx2, sentence2 in enumerate(self.article2.sentences):
                pairs.append((self.similarity_matrix[idx1, idx2].item(), sentence1, sentence2))
            pairs.sort(key=lambda pair: pair[0], reverse=True)

            candidates.extend(pairs[:self.__pair_number])
            pairs.clear()
        return candidates
