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
Input: python3.7 senalign.py -l language_1 language_2 -i input_1.txt input_2.txt -o output_1.txt output_2.txt
Inwhich:
	language_1: language 1
	language_2: language 2
	input_1.txt: input text file of language 1
	input_2.txt: input text file of language 2
	output_1.txt: output text file of language 1
	output_2.txt: output text file of language 2
	
Eg: python3.7 senalign.py -l km vi -i km.txt vi.txt -o out_km.txt out_vi.txt
```

# Note
```
CURRENTLY SUPPORTED LANGUAGUE: km , vi 
```
