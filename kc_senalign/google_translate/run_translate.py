from google_translate.client import Translator
import time
import os
import unicodedata

translator = Translator()

# def translate_file(file_in , file_out , src_lg = 'km' , dest_lg = 'vi'):
#     print('translating from ', src_lg , 'to  ', dest_lg)
#     with open(file_out,'a+', encoding='utf-8') as f:
#         f.seek(0)
#         n_processed = len(f.readlines())
#         print(n_processed,"sens have been processed")
#         f.seek(0, os.SEEK_END)
#         for i,line in enumerate(open(file_in, encoding='utf-8')):
#             time.sleep(2)
#             if i < n_processed:
#                 continue
#             line = line.strip()
#             doc_id, doc_content = line.split("<<<f>>>")[0] , line.split("<<<f>>>")[1]
#             ### Remove emoji icon that's not accepted by ChromeDriver
#             doc_content = ''.join(c for c in unicodedata.normalize('NFC', doc_content) if c <= '\uFFFF')
#             if doc_content != '':
#                 print('process sentence ', i , len(doc_content))
#                 f.write(doc_id + "<<<f>>>" +translator.translate(doc_content , src = src_lg , dest=dest_lg)+'\n')
def translate_str(str_in , src_lg = 'km' , dest_lg = 'vi'):
	# print(str_in)
	# with open(file_out,'w', encoding='utf-8') as f:
		# for i,line in enumerate(open(file_in, encoding='utf-8')):
	time.sleep(1)
			# line = line.strip()
	# doc_id, doc_content = line.split("<<<f>>>")[0] , line.split("<<<f>>>")[1]
			### Remove emoji icon that's not accepted by ChromeDriver
			# doc_content = ''.join(c for c in unicodedata.normalize('NFC', doc_content) if c <= '\uFFFF')
			# if doc_content != '':
			# 	print('translating sentence ', i , len(doc_content))
	return translator.translate(str_in , src = src_lg , dest=dest_lg)

def translate_file(file_in , file_out , src_lg = 'km' , dest_lg = 'vi'):
	print('translating from ', src_lg , 'to  ', dest_lg)
	with open(file_out,'w', encoding='utf-8') as f:
		for i,line in enumerate(open(file_in, encoding='utf-8')):
			time.sleep(1)
			line = line.strip()
			doc_id, doc_content = line.split("<<<f>>>")[0] , line.split("<<<f>>>")[1]
			### Remove emoji icon that's not accepted by ChromeDriver
			doc_content = ''.join(c for c in unicodedata.normalize('NFC', doc_content) if c <= '\uFFFF')
			if doc_content != '':
				print('translating sentence ', i , len(doc_content))
				f.write(doc_id + "<<<f>>>" +translator.translate(doc_content , src = src_lg , dest=dest_lg)+'\n')
