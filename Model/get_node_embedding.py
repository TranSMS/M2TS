from bert_serving.client import BertClient
import numpy as np
import torch
import scipy.sparse as sp
# from get_node import ast_node
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# max_node = 22


def get_embed(ast_file, max_node):
    X = []

    file = open(ast_file, 'r', encoding='utf-8')
    papers = []
    for line in file.readlines():
        dic = json.loads(line)

        papers.append(dic)
    for ast in papers:
        val = []
        for b in ast:
            if 'value' in b.keys():
                val.append(b['value'])
            else:
                val.append('')
            ty = [b['type'] for b in ast]
        node = []
        for i in range(0, len(ty)):
            if val[i] != '':
                node.append(ty[i] + '_' +val[i])

            else:
                node.append(ty[i])
        bc = BertClient()
        matrix = bc.encode(node)
        matrix = np.array(matrix)
        matrix = sp.csr_matrix(matrix, dtype=np.float32)
        feature = torch.FloatTensor(np.array(matrix.todense()))
        if feature.size(0) > max_node:
            features = feature[0:max_node]
        else:
            features = torch.zeros(max_node, 768)
            for k in range(feature.size(0)):
                features[k] = feature[k]

        X.append(features)

    return X
