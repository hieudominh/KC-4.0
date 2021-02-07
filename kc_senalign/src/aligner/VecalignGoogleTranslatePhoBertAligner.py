import os
import subprocess
import sys

from src.aligner.GoogleTranslatePhoBertAligner import GoogleTranslatePhoBertAligner
from src.article.Article import Article
from src.article.EmbeddableArticle import EmbeddableArticle
from src.utils.FileHandler import FileHandler

VECALIGN_TMP_DIR = os.path.join('tmp', 'vecalign')
OVERLAP_PROGRAM = os.path.join('lib', 'vecalign', 'overlap.py')
NUM_OVERLAPS = 4


class VecalignGoogleTranslatePhoBertAligner(GoogleTranslatePhoBertAligner):

    def __init__(self, device):
        super().__init__(device)
        self.file_handler = FileHandler()

    def _process_alignment(self, article1: Article, article2: Article, translation1: EmbeddableArticle,
                           translation2: EmbeddableArticle):
        src_file_path = self.__create_raw_file(article1)
        src_overlap_file_path, src_emb_file_path = self.__create_overlaps(article1, translation1)

        tgt_file_path = self.__create_raw_file(article2)
        tgt_overlap_file_path, tgt_emb_file_path = self.__create_overlaps(article2, translation2)
        self.mapper.set_params(src_file_path, tgt_file_path, src_overlap_file_path, src_emb_file_path,
                               tgt_overlap_file_path, tgt_emb_file_path)
        mappings = self.mapper.map_sentences()

        alignment_results = []
        for mapping in mappings:
            cost, src_ids, tgt_ids = mapping
            alignment_results.append((cost,
                                      ' '.join([article1.sentences[idx] for idx in src_ids]),
                                      ' '.join([article2.sentences[idx] for idx in tgt_ids])
                                      ))

        return alignment_results

    def __create_overlaps(self, article, translation):
        overlap_file_path = self.__generate_overlap_file(article, translation)
        overlap_embeddings = self.__generate_overlap_embeddings(overlap_file_path)
        src_emb_file_path = os.path.join(VECALIGN_TMP_DIR, 'overlap_emb_' + article.langid + '.bin')
        self.file_handler.to_binary_file(src_emb_file_path, overlap_embeddings)
        return overlap_file_path, src_emb_file_path

    def __generate_overlap_embeddings(self, overlap_file_path):
        overlaps = self.file_handler.from_file(overlap_file_path)
        overlap_embeddings = []
        for line in overlaps:
            overlap_embeddings.append(self._embedder.embed_text(line).numpy())
        return overlap_embeddings

    def __generate_overlap_file(self, article, translation):
        trans_file_path = os.path.join(VECALIGN_TMP_DIR, 'trans_' + article.langid + '.txt')
        self.file_handler.to_file(trans_file_path, translation.sentences)
        overlap_file_path = os.path.join(VECALIGN_TMP_DIR, 'overlap_' + article.langid + '.txt')

        result = subprocess.run(
            [sys.executable,
             OVERLAP_PROGRAM,
             '-i', trans_file_path,
             '-o', overlap_file_path
             ], capture_output=True, text=True, timeout=60
        )
        print(result.stderr)
        # if result.stderr is not None or result.stderr != '':
        #     raise Exception(result.stderr)

        return overlap_file_path

    def __create_raw_file(self, article):
        file_path = os.path.join(VECALIGN_TMP_DIR, article.langid + '.txt')
        self.file_handler.to_file(file_path, article.sentences)
        return file_path
