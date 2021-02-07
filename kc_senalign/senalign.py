import argparse
import sys

import torch

from src.aligner.MnGoogleTranslatePhoBertAligner import MnGoogleTranslatePhoBertAligner
from src.aligner.VecalignGoogleTranslatePhoBertAligner import VecalignGoogleTranslatePhoBertAligner
from src.mapper.OneThresholdMapper import OneThresholdEvaluator
from src.mapper.TopPairMapper import TopPairEvaluator
from src.mapper.VecalignMapper import VecalignMapper


def input_file(args):
    lang_1 = ''
    lang_2 = ''
    inputfile = ''
    inputfile_1 = ''
    inputfile_2 = ''
    outputfile = ''
    # outputfile_1 = ''
    # outputfile_2 = ''
    for arg in args:
        if arg[0] == 'language':
            lang_1 = arg[1]
            lang_2 = 'vi'
            if lang_1 != 'km': 
                if lang_1 != 'zh':
                    print('language not supported')
                    sys.exit()
        elif arg[0] == 'source':
            inputfile_2 = arg[1]
        elif arg[0] == 'target':
            inputfile_1 = arg[1]
        elif arg[0] == 'output':
            outputfile = arg[1]
        elif arg[0] == 'threshold':
            threshold = arg[1]
        elif arg[0] == 'maxpair':
            maxpair = arg[1]
    try:
        with open(inputfile_1) as file_in:
            string_1 = file_in.read()
    except Exception:
        print('File not found: ', string_1)
        sys.exit()

    try:
        with open(inputfile_2) as file_in:
            string_2 = file_in.read()
    except Exception:
        print('File not found: ', string_2)
        sys.exit()

    return lang_1, lang_2, string_1, string_2, outputfile, threshold, maxpair


def out_to_file(lang_1, lang_2, outputfile, sentence_pairs):
    # print((sentence_pairs))
    f = open(outputfile, 'w+')

    for i in range(len(sentence_pairs)):
        pair = sentence_pairs[i]
        # print(pair)
        f.write(str(pair[0]) + "\t" + pair[2] + "\t" + pair[1] + "\n")

    f.close()


def main():
    parser = argparse.ArgumentParser(
        description='-s <source_language_file> -t <target_language_file> -lang <target_language> -o <output_file> -thres <threshold> -pair <maxpair>'
    )
    parser.add_argument('-s', '--source', nargs='?', help='Input a source language file', required=True)
    parser.add_argument('-t', '--target', nargs='?', help='Input a target language file', required=True)
    parser.add_argument('-lang', '--language', nargs='?', help='Input a target language', required=True)
    parser.add_argument('-o', '--output', nargs='?', help='Input output file', required=True)
    parser.add_argument('-thres', '--threshold', nargs='?', const=0.6, type=float, default=0.6)
    parser.add_argument('-pair', '--maxpair', nargs='?', const=0,default=0,type=int, required=False)
    parser.add_argument('--vecalign', nargs='?', const='phobert', type=str, help='To use vecalign with PhoBERT embeddings and Machine translation',  required=False)

    args = parser.parse_args()
    args_list = args._get_kwargs()
    lang_1, lang_2, string_1, string_2, outputfile, threshold, maxpair = input_file(args_list)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.vecalign == 'phobert':
        aligner = VecalignGoogleTranslatePhoBertAligner(device)
        alignment_max_size = 3
        aligner.set_mapper(VecalignMapper(alignment_max_size))
    else:
        aligner = MnGoogleTranslatePhoBertAligner(device)
        if (maxpair > 0):
            aligner.set_mapper(TopPairEvaluator(maxpair))
        else:
            aligner.set_mapper(OneThresholdEvaluator(threshold))

    aligner.set_article_langid_pair(lang_1, lang_2)
    aligner.set_article_pair(string_1, string_2)
    sentence_pairs = aligner.align()
    out_to_file(lang_1, lang_2, outputfile, sentence_pairs)

    aligner.stop()
    sys.exit()


if __name__ == "__main__":
    main()
