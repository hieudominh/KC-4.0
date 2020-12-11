import argparse
import re
import sys
import time
from typing import List

import torch
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
rdrsegmenter = VnCoreNLP("./lib/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
print(rdrsegmenter)
phobert = AutoModel.from_pretrained("vinai/phobert-base", output_hidden_states=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
cos = torch.nn.CosineSimilarity(dim=0)
driver = webdriver.Chrome('./lib/webdriver/google-chrome/chromedriver')
KM_SENTENCE_END = r"(?<!\w[\.៕។]\w[\.៕។])(?<![A-Z][a-z][\.៕។])(?<=[\.៕។]|\?|\!)\s"
# src_text = driver.find_element_by_xpath('//textarea[@id="source"]')
prev_text = ''


def translate(text):
    # print(text)
    global prev_text

    def find_trans_box(driver):
        element = driver.find_element_by_xpath('//span[@jsname="W297wb"]')
        # print(element.text)
        if element and element.text == prev_text:
            # print("bug")
            return False
        if element and element.text != prev_text:
            return element
        else:
            return False

    def find_src_box(driver):
        element = driver.find_element_by_xpath('//textarea[@class="er8xn"]')
        if element:
            return element
        else:
            return False

    src_text = WebDriverWait(driver, 50).until(find_src_box)
    src_text.clear()
    src_text.send_keys(text)
    time.sleep(1)
    # WebDriverWait(driver, 50).until(EC.presence_of_element_located((By.XPATH, '//div[@class="text-wrap tlid-copy-target"]')))
    dst_text = WebDriverWait(driver, 50).until(find_trans_box)
    time.sleep(1)
    # dst_text = driver.find_element_by_xpath('//div[@class="text-wrap tlid-copy-target"]')
    prev_text = dst_text.text
    return dst_text.text


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
    '''
  Concat if segmented into sentences
  '''
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


def detect_sentences(text_list: List[str], end_signs: str):
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


def process(lang1, lang2, string1, string2):
    km_translate = []
    if lang1 == 'km':
        driver.get('https://translate.google.com/?sl=' + lang1 + '&tl=' + lang2 + '&op=translate')
        km_sentences = detect_sentences([string1], KM_SENTENCE_END)
        for km_sentence in km_sentences:
            km_translate.append(translate(km_sentence))
        src_vn = string2
    elif lang2 == 'km':
        km_sentences = detect_sentences([string2], KM_SENTENCE_END)
        for km_sentence in km_sentences:
            km_translate.append(translate(km_sentence))
        src_vn = string1

    # print('\nvn\n')
    vn_segment, vn = embed_sentence_with_phobert(text=src_vn, segmenter=rdrsegmenter, tokenizer=tokenizer,
                                                 model=phobert, device=device)
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
            sentence_pairs.append((sim.item(), km_sentences[j], ' '.join(vn_segment[i]).replace("_", " ")))

    return sentence_pairs


def input_string(args):
    lang_1 = ''
    lang_2 = ''
    string_1 = ''
    string_2 = ''
    for arg in args:
        if arg[0] == 'lang':
            lang = arg[1]
            if len(lang) != 2:
                print('senalign.py -lang <lang1>,<lang2> -i <inputfile1>,<inputfile2> -o <outputfile1>,<outputfile2>')
                sys.exit()
            else:
                if 'km' in lang:
                    if 'vi' in lang:
                        lang_1 = lang[0]
                        lang_2 = lang[1]
                    else:
                        print('Currently support: km, vi')
                else:
                    print('Currently support: km, vi')
        elif arg[0] == 'string':
            inputstring = arg[1]
            if len(inputstring) != 2:
                print('senalign.py -lang <lang1>,<lang2> -i <inputfile1>,<inputfile2> -o <outputfile1>,<outputfile2>')
                sys.exit()
            else:
                string_1 = inputstring[0]
                string_2 = inputstring[1]
        elif arg[0] == 'output':
            outputfile = arg[1]
            if len(outputfile) != 2:
                print('senalign.py -lang <lang1>,<lang2> -i <inputfile1>,<inputfile2> -o <outputfile1>,<outputfile2>')
                sys.exit()
            else:
                outputfile_1 = outputfile[0]
                outputfile_2 = outputfile[1]
    return lang_1, lang_2, string_1, string_2, outputfile_1, outputfile_2


def input_file(args):
    lang_1 = ''
    lang_2 = ''
    inputfile = ''
    inputfile_1 = ''
    inputfile_2 = ''
    outputfile = ''
    outputfile_1 = ''
    outputfile_2 = ''
    for arg in args:
        if arg[0] == 'lang':
            lang = arg[1]
            if len(lang) != 2:
                print('senalign.py -lang <lang1>,<lang2> -i <inputfile1>,<inputfile2> -o <outputfile1>,<outputfile2>')
                sys.exit()
            else:
                if 'km' in lang:
                    if 'vi' in lang:
                        lang_1 = lang[0]
                        lang_2 = lang[1]
                    else:
                        print('Currently support: km, vi')
                else:
                    print('Currently support: km, vi')
        elif arg[0] == 'input':
            inputfile = arg[1]
            if len(inputfile) != 2:
                print('senalign.py -lang <lang1>,<lang2> -i <inputfile1>,<inputfile2> -o <outputfile1>,<outputfile2>')
                sys.exit()
            else:
                inputfile_1 = inputfile[0]
                inputfile_2 = inputfile[1]
        elif arg[0] == 'output':
            outputfile = arg[1]
            if len(outputfile) != 2:
                print('senalign.py -lang <lang1>,<lang2> -i <inputfile1>,<inputfile2> -o <outputfile1>,<outputfile2>')
                sys.exit()
            else:
                outputfile_1 = outputfile[0]
                outputfile_2 = outputfile[1]
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

    return lang_1, lang_2, string_1, string_2, outputfile_1, outputfile_2


def out_to_file(lang_1, lang_2, outputfile_1, outputfile_2, sentence_pairs):
    f1 = open(outputfile_1, 'w+')
    f2 = open(outputfile_2, 'w+')

    for i in range(len(sentence_pairs)):
        pair = sentence_pairs[i]
        f1.write(lang_1 + "_" + str(i) + "_" + str(pair[0]) + "\t" + pair[1] + "\n")
        f2.write(lang_2 + "_" + str(i) + "_" + str(pair[0]) + "\t" + pair[2] + "\n")

    f1.close()
    f2.close()


def main():
    parser = argparse.ArgumentParser(
        description='-l <lang1> <lang2> -i <inputfile1> <inputfile2> -o <outputfile1> <outputfile2>' + '//' + ' -l <lang1> <lang2> -s <string1> <string2> -o <outputfile1> <outputfile2>')
    parser.add_argument('-l', '--lang', nargs='+', help='Input 2 language IDs', required=True)
    parser.add_argument('-s', '--string', nargs='+',
                        help='Input 2 strings with respect to the two languages whose IDs are given', required=False)
    parser.add_argument('-i', '--input', nargs='+',
                        help='Input 2 input files with respect to the two languages whose IDs are given',
                        required=False)
    parser.add_argument('-o', '--output', nargs='+',
                        help='Input 2 output file with respect to the two language whose IDs are given', required=True)
    args = parser.parse_args()
    args_list = args._get_kwargs()
    for arg in args_list:
        if arg[0] == 'input':
            if arg[1] != None:
                lang_1, lang_2, string_1, string_2, outputfile_1, outputfile_2 = input_file(args_list)
            else:
                lang_1, lang_2, string_1, string_2, outputfile_1, outputfile_2 = input_string(args_list)

    sentence_pairs = process(lang_1, lang_2, string_1, string_2)
    out_to_file(lang_1, lang_2, outputfile_1, outputfile_2, sentence_pairs)

    sys.exit()

    

if __name__ == "__main__":
    main()
