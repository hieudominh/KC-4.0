# Requirement
```
Google Chrome (optional)
Java 8 or higher
Python 3.7 or higher
Internet connection
```
# Installation:
```
conda activate ./envs

OR

install the following libraries:
(py)torch = 1.5.0
selenium = 3.141.0
vncorenlp = 1.0.3
transformers = 3.5.1
numpy = 1.19.2
cython
laserembeddings
```

# How to run
```
Input: python3 senalign.py -s source.txt -t target.txt -o output.txt -lang lang [-thres threshold, -pair k, --vecalign embedder]
Inwhich:
	source.txt: text of source language
	target.txt: text of target language
	output.txt: output text
	lang: target language
	threshold: threshold of similarity (default : 0.6)
    maxpair: return best k candidates for each sentence (not work with threshold option)
    vecalign: choose the embedder for sentences (currently only support phobert)
Eg: python3 senalign.py -s vi.txt -t km.txt -o vi_km.txt -lang km -thres 0.6
    python3 senalign.py -s vi.txt -t km.txt -o vi_km.txt -lang km -pair 5
    python3 senalign.py -s vi.txt -t km.txt -o vi_km.txt -lang km --vecalign phobert
```

# Note
```
CURRENTLY SUPPORTED LANGUAGUE: km,zh
Suggested threshold: 0.7
```
