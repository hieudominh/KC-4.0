import unittest

import torch
import os

from src.comparator.CosineSimilarityComparator import CosineSimilarityComparator
from src.embeder.PhoBert import PhoBert
from src.sentence.EmbeddableSentence import EmbeddableSentence


class TestPhoBert(unittest.TestCase):
    def test_embed(self):
        # self.assertEqual(True, False)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        sentence1 = EmbeddableSentence(
            'Hà Nội và Thành phố Hồ Chí Minh đã lọt vào danh sách 10 thành phố mới nổi về gia công phần mềm .', 'vi')
        sentence2 = EmbeddableSentence(
            'Đáng chú ý hai thành phố Hà Nội và Tp.HCM đã lọt vào danh sách 10 thành phố mới nổi về gia công phần mềm .',
            'vi')
        sentence3 = EmbeddableSentence(
            'Thủ tướng cho rằng công nghệ thông tin là một lợi thế phát triển đặc biệt của Việt Nam trên nền tảng nguồn nhân lực trẻ dồi dào và được đào tạo cơ bản .',
            'vi')

        os.chdir('../../')

        phobert = PhoBert(device)

        sentence1.to_embedding(phobert)
        sentence2.to_embedding(phobert)
        sentence3.to_embedding(phobert)

        comparator = CosineSimilarityComparator()

        print('1 vs 2 :', sentence1.cmp(sentence2, comparator))
        print('1 vs 3 :', sentence1.cmp(sentence3, comparator))
        print('3 vs 2 :', sentence3.cmp(sentence2, comparator))


if __name__ == '__main__':
    unittest.main()
