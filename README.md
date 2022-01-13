# M2TS: Multi-scale Multi-modal Approach Based on Transformer for Source Code Summarization
The source code, datasets and results for M2TS.
# Datasets
In M2TS, we use three large-scale datasets for experiments, including two Java and one Python datasets. In data file, we give the three datasets, which obtain from following paper.
## JAH(Java Hu) dataset
* paper: https://arxiv.org/abs/1707.02275
* data: https://github.com/EdinburghNLP/code-docstring-corpus
## PYB(Python Barone) dataset
* paper: https://xin-xia.github.io/publication/ijcai18.pdf
* data: https://github.com/xing-hu/TL-CodeSum
## JAL(Java Leclair) dataset
* paper: https://arxiv.org/pdf/1904.02660.pdf
* data: http://leclair.tech/data/funcom/
# Data preprocessing
M2TS uses ASTs and source code modalities, which uses the JDK compiler to parse java methods as ASTs, and the Treelib toolkit to prase Python functions as ASTs. In addition, before embedding ASTs, we use BERT pre-training to embed the information of nodes. 
## Get ASTs
In data-pre file, the get_ast.py generates ASTs for two Java datasets and python_ast.py generates ASTs for Python functions. 
Command: python3 source.code ast.json
### BERT Pre-training
Get here to Install the server and client, detail in [here](https://github.com/hanxiao/bert-as-service)  
`pip install bert-serving-server`  
`pip install bert-serving-client`
# Train-Test
In M2TS_model file, the run.py train the model and run2.py is the model without multi-modal fusion module which can train and test the M2TS. 
Command: Directly run run.py
# Requirements
pytorch 1.7.1  
bert-serving-client 1.10.0  
bert-serving-server 1.10.0  
javalang 0.13.0  
nltk 3.5  
networkx 2.5  
scipy 1.1.0  
treelib 1.6.1
# Results
In result file, we give the testing results on three datasets. The java_pre.txt is the generated summaries for JAH dataset.
