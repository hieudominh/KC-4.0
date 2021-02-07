import torch

from src.aligner.GoogleTranslatePhoBertAligner import GoogleTranslatePhoBertAligner
from src.article.Article import Article
from src.article.EmbeddableArticle import EmbeddableArticle
from src.comparator.CosineSimilarityComparator import CosineSimilarityComparator
from src.mapper.MnEmbeddingMapper import MnEmbeddingMapper


class MnGoogleTranslatePhoBertAligner(GoogleTranslatePhoBertAligner):
    def __init__(self, device):
        super().__init__(device)
        self.__comparator = CosineSimilarityComparator()

    def set_mapper(self, mapper: MnEmbeddingMapper):
        self.mapper = mapper

    def __cmp(self, translation1: EmbeddableArticle, translation2: EmbeddableArticle):
        similarity_matrix = torch.zeros((len(translation1.sentences), len(translation2.sentences)))
        # print(translation1.sentences)
        # print(translation2.sentences)
        for idx1, sentence1 in enumerate(translation1.embedded_sentences):
            for idx2, sentence2 in enumerate(translation2.embedded_sentences):
                # print(sentence1,'\n',sentence2,'\n\n')
                similarity_matrix[idx1, idx2] = sentence1.cmp(sentence2, self.__comparator)
        return similarity_matrix

    def _process_alignment(self, article1: Article, article2: Article, translation1: EmbeddableArticle, translation2: EmbeddableArticle):
        # embed sentences
        translation1.embed_sentences(self._embedder)
        translation2.embed_sentences(self._embedder)

        # compare embeddings of sentence pairs
        similarity_matrix = self.__cmp(translation1, translation2)
        # mapping
        self.mapper.set_params(article1, article2, similarity_matrix)
        result = self.mapper.map_sentences()
        return result
