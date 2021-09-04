# M2TS: Multi-scale Multi-modal Approach Based on Transformer for Source Code Summarization
The source code, datasets and results for M2TS.
# Datasets
In M2TS, we use two large-scale datasets for experiments.
## java_data
* paper: https://arxiv.org/abs/1707.02275
* data: https://github.com/EdinburghNLP/code-docstring-corpus
## python_data
* paper: https://xin-xia.github.io/publication/ijcai18.pdf
* data: https://github.com/xing-hu/TL-CodeSum
# Data preprocessing
M2TS uses ASTs and source code modalities, which uses the JDK compiler to parse java methods as ASTs, and the Treelib toolkit to prase Python functions as ASTs. In addition, before embedding ASTs, we use BERT pre-training to embed the information of nodes. 
## Get ASTs
In data-pre file, the get_ast.py generates ASTs for Java method and get_python.py generates ASTs for Python functions. 
Command: python3 source.code ast.json
### BERT Pre-training
Get here to Install the server and client via pip：
detail in here: https://github.com/hanxiao/bert-as-service
## Train-model
In model_train file, the run.py train the model and run2.py is the model without multi-modal fusion module which can train and test the M2TS. 
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
## Results
Results of the M2TS projection
