# M2TS: Multi-scale Multi-modal Approach Based on Transformer for Source Code Summarization
We public the source code, datasets and results for the M2TS.
# Datasets
In the M2TS, we use three large-scale datasets for experiments, including two Java and one Python datasets. In data file, we give the three datasets, which obtain from following paper. If you want to train the model, you must download the datasets.
## PYB(Python Barone) dataset
* paper: https://arxiv.org/abs/1707.02275
* data: https://github.com/EdinburghNLP/code-docstring-corpus
## JAH(Java Hu) dataset
* paper: https://xin-xia.github.io/publication/ijcai18.pdf
* data: https://github.com/xing-hu/TL-CodeSum
## JAL(Java Leclair) dataset
* paper: https://arxiv.org/pdf/1904.02660.pdf
* data: http://leclair.tech/data/funcom/
# Data preprocessing
M2TS uses ASTs and source code modalities, which uses the [JDK](http://www.eclipse.org/jdt/) compiler to parse java methods as ASTs, and the [Treelib](https://treelib.readthedocs.io/en/latest/) toolkit to prase Python functions as ASTs. In addition, before embedding ASTs, we use `BERT method` to embed the information of nodes. 
## Get ASTs
In Data-pre file, the `java_get_ast.py` generates ASTs for two Java datasets and `python_get_ast.py` generates ASTs for Python functions. You can run the following command：
```
python3 source.code ast.json
```
### BERT embedding
Get here to Install the server and client, detail in [here](https://github.com/hanxiao/bert-as-service)  
```
pip install bert-serving-server  
pip install bert-serving-client
```
# Train-Test
In Model file, the `_main_.py` enables to train the model. We evaluate the quality of the generated summaries using four evaluation metrics.
Train and test model:  
```
Python3 _main_.py
```
The nlg-eval can be set up in the following way, detail in [here](https://github.com/Maluuba/nlg-eval).  
Install Java 1.8.0 (or higher).  
Install the Python dependencies, run:
```
pip install git+https://github.com/Maluuba/nlg-eval.git@master
```
# Requirements
If you want to run the model, you will need to install the following packages.  
```
pytorch 1.7.1  
bert-serving-client 1.10.0  
bert-serving-server 1.10.0  
javalang 0.13.0  
nltk 3.5  
networkx 2.5  
scipy 1.1.0  
treelib 1.6.1
```
# Results
In result file, we give the testing results on the datasets. The `Python_pre.txt` is the generated summaries for PYB dataset.
