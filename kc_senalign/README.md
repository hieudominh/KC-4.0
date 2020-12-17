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
```

# How to run
```
Input: python3 senalign.py -s source.txt -t target.txt -o output.txt -lang lang -thres threshold
Inwhich:
	source.txt: text of source language
	target.txt: text of target language
	output.txt: output text
	lang: target language
	threshold: threshold of similarity (default : 0.6)
	
Eg: python3 senalign.py -s vi.txt -t km.txt -o vi_km.txt -lang km -thres 0.8
```

# Note
```
CURRENTLY SUPPORTED LANGUAGUE: km 
Suggested threshold: 0.6
```
