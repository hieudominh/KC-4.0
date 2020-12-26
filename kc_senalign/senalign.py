import argparse
import re
import sys
import time
from typing import List
from google_translate.run_translate import *
from laserembeddings import Laser
import numpy as np
import torch
from selenium import webdriver
# from selenium.webdriver.support.ui import WebDriverWait
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP

laser = Laser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
rdrsegmenter = VnCoreNLP("./lib/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
print(rdrsegmenter)
phobert = AutoModel.from_pretrained("vinai/phobert-base", output_hidden_states=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
cos = torch.nn.CosineSimilarity(dim=0)
# driver = webdriver.Chrome('./lib/webdriver/google-chrome/chromedriver')
KM_SENTENCE_END = r"(?<!\w[\.៕។]\w[\.៕។])(?<![A-Z][a-z][\.៕។])(?<=[\.៕។]|\?|\!)\s"
# src_text = driver.find_element_by_xpath('//textarea[@id="source"]')
prev_text = ''


def translate(str_in, src_lg, dest_lg):
    return translate_str(str_in=str_in, src_lg=src_lg, dest_lg=dest_lg)


# def translate(text):
#     # print(text)
#     global prev_text

#     def find_trans_box(driver):
#         element = driver.find_element_by_xpath('//span[@jsname="W297wb"]')
#         # print(element.text)
#         if element and element.text == prev_text:
#             # print("bug")
#             return False
#         if element and element.text != prev_text:
#             return element
#         else:
#             return False

#     def find_src_box(driver):
#         element = driver.find_element_by_xpath('//textarea[@class="er8xn"]')
#         if element:
#             return element
#         else:
#             return False

#     src_text = WebDriverWait(driver, 50).until(find_src_box)
#     src_text.clear()
#     src_text.send_keys(text)
#     time.sleep(1)
#     # WebDriverWait(driver, 50).until(EC.presence_of_element_located((By.XPATH, '//div[@class="text-wrap tlid-copy-target"]')))
#     dst_text = WebDriverWait(driver, 50).until(find_trans_box)
#     time.sleep(1)
#     # dst_text = driver.find_element_by_xpath('//div[@class="text-wrap tlid-copy-target"]')
#     prev_text = dst_text.text
#     return dst_text.text

def embed_sentences_with_laser(text_list: List[str], lang: str) -> List[torch.tensor]:
    embeddings = laser.embed_sentences(text_list, lang)
    embeddings = [torch.from_numpy(embedding) for embedding in embeddings]
    return embeddings


def embed_sentence_with_phobert(text: str, segmenter, tokenizer, model, device):
    try:
        segments = segmenter.tokenize(text)
        lines = [' '.join(segment) for segment in segments]
        #
        # print('to be embeded: ', lines)
        input_ids_list = [tokenizer.encode(line, return_tensors="pt").to(device) for line in lines]
        #
        model.to(device)
        with torch.no_grad():
            features_list = [model(input_ids) for input_ids in input_ids_list]
        hidden_states_list = [features[2] for features in features_list]
        last_four_layers_list = []
        for hidden_states in hidden_states_list:
            last_four_layers_list.append([hidden_states[i] for i in (-1, -2, -3, -4)])
        cat_hidden_states_list = [torch.cat(tuple(last_four_layers), dim=-1) for last_four_layers in
                                  last_four_layers_list]
        return segments, [torch.mean(cat_hidden_states, dim=1).squeeze() for cat_hidden_states in
                          cat_hidden_states_list]
    except Exception as e:
        print(e)
        return None


def embed_concatenated_sentence_with_phobert(text: str, segmenter, tokenizer, model, device):
    """
    Concat if segmented into sentences
    """
    try:
        segments = segmenter.tokenize(text)
        lines = [' '.join(segment) for segment in segments]
        #
        segment_list = []
        for segment in segments:
            segment_list += segment
        lines = ' '.join(lines)
        # print('to be concat & embeded: ', lines)
        input_ids_list = [tokenizer.encode(lines, return_tensors="pt").to(device)]
        #
        model.to(device)
        with torch.no_grad():
            features_list = [model(input_ids) for input_ids in input_ids_list]
        hidden_states_list = [features[2] for features in features_list]
        last_four_layers_list = []
        for hidden_states in hidden_states_list:
            last_four_layers_list.append([hidden_states[i] for i in (-1, -2, -3, -4)])
        cat_hidden_states_list = [torch.cat(tuple(last_four_layers), dim=-1) for last_four_layers in
                                  last_four_layers_list]
        return segment_list, [torch.mean(cat_hidden_states, dim=1).squeeze() for cat_hidden_states in
                              cat_hidden_states_list]
    except Exception as e:
        print(e)
        return None


def detect_title(text: str) -> List[str]:
    '''
    Detect title in a document
    :param text: a document
    :return: List[title, contents] or List[contents]
    '''
    if text is None:
        raise TypeError
    return text.split("\n", 1)


def extract_vi_sentences(text_list: List[str], segmenter):
    sentences = []
    for text in text_list:
        segments = segmenter.tokenize(text)
        lines = [' '.join(segment).replace("_", " ") for segment in segments]
        sentences += lines
    return sentences


def detect_km_sentences(text_list: List[str], end_signs: str):
    # if(not isinstance(text_list, list)):
    #     raise TypeError
    # elif(len(text_list) == 0):
    #     raise ValueError
    '''
    For Khmer only
    '''

    def split_with_re(sep: str, lines: list):
        strings = []
        for line in lines:
            line.replace("\n", " ")
            strs = re.split(sep, line)
            # print(strs)
            try:
                strs.remove('')
            except ValueError:
                pass
            strings += strs
        return strings

    sentences = split_with_re(end_signs, text_list)
    return sentences


def process3(lang1, lang2, string1, string2, maxpair):
    """
   Laser with maxpairs
   """
    km_sentences = []

    if lang1 == 'km':
        parts = detect_title(string1)
        if len(parts) == 1:
            contents = parts[0]
        elif len(parts) == 2:
            km_sentences = [parts[0]]
            contents = parts[1]
        else:
            raise TypeError
        km_sentences += detect_km_sentences([contents], KM_SENTENCE_END)
        src_vn = string2
    elif lang2 == 'km':
        parts = detect_title(string2)
        if len(parts) == 1:
            contents = parts[0]
        elif len(parts) == 2:
            km_sentences = [parts[0]]
            contents = parts[1]
        else:
            raise TypeError
        km_sentences += detect_km_sentences([contents], KM_SENTENCE_END)
        src_vn = string1
    # print(km_sentences)
    km = embed_sentences_with_laser(km_sentences, "km")

    parts = detect_title(src_vn)
    vi_sentences = []
    if len(parts) == 1:
        contents = parts[0]
    elif len(parts) == 2:
        vi_sentences += parts[0]
        contents = parts[1]
    else:
        raise TypeError

    vi_sentences = extract_vi_sentences([contents], rdrsegmenter)
    # print(vi_sentences)
    vi = embed_sentences_with_laser(vi_sentences, "vi")

    sentence_pairs = []

    for i in range(len(vi)):
        pairs = []
        for j in range(len(km)):
            sim = cos(vi[i], km[j])
            pairs.append((sim.item(), km_sentences[j], vi_sentences[i]))
        pairs.sort(key=lambda pair: pair[0], reverse=True)
        sentence_pairs += pairs[:maxpair]
        del pairs
    return sentence_pairs


def process2(lang1, lang2, string1, string2, threshold):
    """
    With LASER
    """
    km_sentences = []

    if lang1 == 'km':
        parts = detect_title(string1)
        if len(parts) == 1:
            contents = parts[0]
        elif len(parts) == 2:
            km_sentences = [parts[0]]
            contents = parts[1]
        else:
            raise TypeError
        km_sentences += detect_km_sentences([contents], KM_SENTENCE_END)
        src_vn = string2
    elif lang2 == 'km':
        parts = detect_title(string2)
        if len(parts) == 1:
            contents = parts[0]
        elif len(parts) == 2:
            km_sentences = [parts[0]]
            contents = parts[1]
        else:
            raise TypeError
        km_sentences += detect_km_sentences([contents], KM_SENTENCE_END)
        src_vn = string1
    # print(km_sentences)
    km = embed_sentences_with_laser(km_sentences, "km")

    parts = detect_title(src_vn)
    vi_sentences = []
    if len(parts) == 1:
        contents = parts[0]
    elif len(parts) == 2:
        vi_sentences += parts[0]
        contents = parts[1]
    else:
        raise TypeError

    vi_sentences = extract_vi_sentences([contents], rdrsegmenter)
    # print(vi_sentences)
    vi = embed_sentences_with_laser(vi_sentences, "vi")

    sentence_pairs = []

    for i in range(len(vi)):
        for j in range(len(km)):
            sim = cos(vi[i], km[j])
            if sim.item() > threshold:
                sentence_pairs.append((sim.item(), km_sentences[j], vi_sentences[i]))
    return sentence_pairs


def process(lang1, lang2, string1, string2, threshold):
    """
    With PhoBert and Google Translate
    """
    km_translate = []
    km_sentences = []

    if lang1 == 'km':
        # driver.get('https://translate.google.com/?sl=' + lang1 + '&tl=' + lang2 + '&op=translate')
        parts = detect_title(string1)
        if len(parts) == 1:
            contents = parts[0]
        elif len(parts) == 2:
            km_sentences = [parts[0]]
            contents = parts[1]
        else:
            raise TypeError
        km_sentences += detect_km_sentences([contents], KM_SENTENCE_END)
        for km_sentence in km_sentences:
            km_translate.append(translate(km_sentence, lang1, lang2))
        src_vn = string2
    elif lang2 == 'km':
        # driver.get('https://translate.google.com/?sl=' + lang2 + '&tl=' + lang1 + '&op=translate')
        parts = detect_title(string2)
        if len(parts) == 1:
            contents = parts[0]
        elif len(parts) == 2:
            km_sentences = [parts[0]]
            contents = parts[1]
        else:
            raise TypeError
        km_sentences += detect_km_sentences([contents], KM_SENTENCE_END)
        for km_sentence in km_sentences:
            km_translate.append(translate(km_sentence, lang2, lang1))
        src_vn = string1
    # print('\nvn\n')
    # print(km_sentences)
    parts = detect_title(src_vn)
    if len(parts) == 1:
        contents = parts[0]
    elif len(parts) == 2:
        title = parts[0]
        contents = parts[1]
    else:
        raise TypeError

    vn_segment, vn = embed_sentence_with_phobert(text=contents, segmenter=rdrsegmenter, tokenizer=tokenizer,
                                                 model=phobert, device=device)
    vn_title_segment, vn_title = embed_sentence_with_phobert(text=title, segmenter=rdrsegmenter, tokenizer=tokenizer,
                                                             model=phobert, device=device)
    vn_segment += vn_title_segment
    vn_title += vn
    # print('\nkm\n')
    km_segment = []
    km = []
    # print(km_translate)
    for km_line in km_translate:
        # print('input: ', km_line)
        seg, vec = (embed_concatenated_sentence_with_phobert(text=km_line,
                                                             segmenter=rdrsegmenter,
                                                             tokenizer=tokenizer,
                                                             model=phobert,
                                                             device=device
                                                             ))
        km.append(vec)
        km_segment.append(seg)

    sentence_pairs = []
    for i in range(len(vn)):
        for j in range(len(km)):
            sim = cos(vn[i], km[j][0])
            # print(sim.item(), vn_segment[i],km_segment[j],km_sentences[j])
            if sim.item() > threshold:
                se = ''
                for vn_seg in vn_segment[i]:
                    if vn_seg == '.':
                        se = se + vn_seg
                    elif vn_seg == ':':
                        se = se + vn_seg
                    elif vn_seg == ',':
                        se = se + vn_seg
                    elif vn_seg == '...':
                        se = se + vn_seg
                    elif vn_seg == ';':
                        se = se + vn_seg
                    else:
                        se = se + ' ' + vn_seg.replace("_", " ")

                # sentence_pairs.append((sim.item(), km_sentences[j], ' '.join(vn_segment[i]).replace("_", " ")))
                sentence_pairs.append((sim.item(), km_sentences[j], se))

    return sentence_pairs

    # def input_string(args):def detect_vi_setences(text_list: List[str], segmenter):
    segments = segmenter.tokenize(text)
    lines = [' '.join(segment) for segment in segments]
    #
    segment_list = []
    for segment in segments:
        segment_list += segment
    return segment_list


#     lang_1 = ''
#     lang_2 = ''
#     string_1 = ''
#     string_2 = ''
#     for arg in args:
#         if arg[0] == 'lang':
#             lang = arg[1]
#             if len(lang) != 2:
#                 print('senalign.py -lang <lang1> <lang2> -i <inputfile1> <inputfile2> -o <outputfile>')  
#                 sys.exit()
#             else:
#                 if 'km' in lang:
#                     if 'vi' in lang:
#                         lang_1 = lang[0]
#                         lang_2 = lang[1]
#                     else:
#                         print('Currently support: km, vi')
#                 else:
#                     print('Currently support: km, vi')
#         elif arg[0] == 'string':python3 senalign.py -s vi.txt -t km.txt -o vi_km.txt -lang km -thres 0.6

#             inputstring = arg[1]
#             if len(inputstring) != 2:
#                 print('senalign.py -lang <lang1> <lang2> -i <inputfile1> <inputfile2> -o <outputfile>')  
#                 sys.exit()
#             else:
#                 string_1 = inputstring[0]
#                 string_2 = inputstring[1]
#         elif arg[0] == 'output':
#             outputfile = arg[1]
#             if len(outputfile) != 1:
#                 print('senalign.py -lang <lang1> <lang2> -i <inputfile1> <inputfile2> -o <outputfile>')  
#                 sys.exit()
#     return lang_1, lang_2, string_1, string_2, outputfile


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
                print('language not supported')
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
    f = open(outputfile, 'w+')

    for i in range(len(sentence_pairs)):
        pair = sentence_pairs[i]
        if lang_1 == 'km':
            f.write(str(pair[0]) + "\t" + pair[2].replace('\n', ' ') + "\t" + pair[1].replace('\n', ' ') + "\n")
        elif lang_2 == 'km':
            f.write(str(pair[0]) + "\t" + pair[1].replace('\n', ' ') + "\t" + pair[2].replace('\n', ' ') + "\n")

    f.close()


def main():
    # parser = argparse.ArgumentParser(
    #     description='-l <lang1> <lang2> -i <inputfile1> <inputfile2> -o <outputfile1> <outputfile2>' + '//' + ' -l <lang1> <lang2> -s <string1> <string2> -o <outputfile1> <outputfile2>')
    # parser.add_argument('-l', '--lang', nargs='+', help='Input 2 language IDs', required=True)
    # parser.add_argument('-s', '--string', nargs='+',
    #                     help='Input 2 strings with respect to the two languages whose IDs are given', required=False)
    # parser.add_argument('-i', '--input', nargs='+',
    #                     help='Input 2 input files with respect to the two languages whose IDs are given',
    #                     required=False)
    # parser.add_argument('-o', '--output', nargs='*',
    #                     help='Input 1 output file with respect to the two language whose IDs are given', required=True)
    # args = parser.parse_args()
    # args_list = args._get_kwargs()
    # for arg in args_list:
    #     if arg[0] == 'input':
    #         if arg[1] != None:
    #             lang_1, lang_2, string_1, string_2, outputfile= input_file(args_list)
    #         else:
    #             lang_1, lang_2, string_1, string_2, outputfile= input_string(args_list)
    parser = argparse.ArgumentParser(
        description='-s <source_language_file> -t <target_language_file> -lang <target_language> -o <output_file> -thres <threshold> -pair <maxpair>'
    )
    parser.add_argument('-s', '--source', nargs='?', help='Input a source language file', required=True)
    parser.add_argument('-t', '--target', nargs='?', help='Input a target language file', required=True)
    parser.add_argument('-lang', '--language', nargs='?', help='Input a target language', required=True)
    parser.add_argument('-o', '--output', nargs='?', help='Input output file', required=True)
    parser.add_argument('-thres', '--threshold', nargs='?', const=0.6, type=float, default=0.6)
    parser.add_argument('-pair', '--maxpair', nargs='?', type=int, required=False)
    args = parser.parse_args()
    args_list = args._get_kwargs()
    lang_1, lang_2, string_1, string_2, outputfile, threshold, maxpair = input_file(args_list)
    if (maxpair > 0):
        sentence_pairs = process3(lang_1, lang_2, string_1, string_2, maxpair)
    else:
        sentence_pairs = process2(lang_1, lang_2, string_1, string_2, threshold)
    out_to_file(lang_1, lang_2, outputfile, sentence_pairs)

    sys.exit()


if __name__ == "__main__":
    main()
