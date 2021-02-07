import os
import subprocess
import sys

from parse import compile
from src.mapper.Mappable import Mappable

VECALIGN_PROGRAM = os.path.join('lib', 'vecalign', 'vecalign.py')
VECALIGN_STDOUT_FORMAT = '[{}]:[{}]:{}'


class VecalignMapper(Mappable):
    def __init__(self, alignment_max_size: int):
        self.alignment_max_size = alignment_max_size
        self.src_file_path = None
        self.tgt_file_path = None
        self.src_overlap_file_path = None
        self.tgt_overlap_file_path = None
        self.src_emb_file_path = None
        self.tgt_emb_file_path = None

    def set_params(self, src_file_path, tgt_file_path, src_overlap_file_path, src_emb_file_path, tgt_overlap_file_path,
                   tgt_emb_file_path):
        self.src_file_path = src_file_path
        self.tgt_file_path = tgt_file_path
        self.src_emb_file_path = src_emb_file_path
        self.src_overlap_file_path = src_overlap_file_path
        self.tgt_overlap_file_path = tgt_overlap_file_path
        self.tgt_emb_file_path = tgt_emb_file_path

    def map_sentences(self) -> list:
        result = subprocess.run(
            [sys.executable, VECALIGN_PROGRAM,
             '--alignment_max_size', str(self.alignment_max_size),
             '--src', self.src_file_path,
             '--tgt', self.tgt_file_path,
             '--src_embed', self.src_overlap_file_path, self.src_emb_file_path,
             '--tgt_embed', self.tgt_overlap_file_path, self.tgt_emb_file_path
             ],
            capture_output=True,
            text=True,
            timeout=60
            )
        # print('\n\nRESULT:\n',result.stdout,'\n\nEND\n\n')
        print(result.stderr)

        # preprocess vecalign output: convert str to structural data
        mappings = []
        mapping_strs = result.stdout.split('\n')

        def parse_idx(ids_str: str):
            return [int(id_str) for id_str in ids_str.split(',')]

        def parse_cost(cost_str: str):
            return float(cost_str)

        parser = compile(VECALIGN_STDOUT_FORMAT)
        for mapping_str in mapping_strs:
            strs = parser.parse(mapping_str)
            if strs is not None:
                src_idx = parse_idx(strs[0])
                tgt_idx = parse_idx(strs[1])
                cost = parse_cost(strs[2])
                mappings.append((cost, src_idx, tgt_idx))
        return mappings
