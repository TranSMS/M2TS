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
Command: python3 source.code ast.json
### BERT Pre-training
Get here to Install the server and client via pip：
detail in here: https://github.com/hanxiao/bert-as-service
## Train-model
        Command: Run run.py
## Results
        Results of the M2TS projection
