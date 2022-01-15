import json
import networkx as nx
import numpy as np
import torch
import scipy.sparse as sp
from scipy import sparse


def read_batchA(ast_file, max_node):
    file = open(ast_file, 'r', encoding='utf-8')
    papers = []
    a1 = []
    aa2 = []
    aa3 = []

    for line in file.readlines():
        dic = json.loads(line)

        papers.append(dic)
        id_ch = {t['id']: t['children'] for t in dic if 'children' in t}
        # print(id_ch)
        # exit()
        edgelist = []
        for id in id_ch:
            for child in id_ch[id]:
                # fo.write(str(id)+'\t'+str(child)+'\n')
                edgelist.append((id, child))

        edgelist = []
        for id in id_ch:
            for child in id_ch[id]:
                # fo.write(str(id)+'\t'+str(child)+'\n')
                edgelist.append((id, child))
        # print(edgelist)
        G = nx.Graph()
        G.add_edges_from(edgelist)

        nx.draw(G, with_labels=True)
        A = np.array(nx.adjacency_matrix(G).todense())
        A1 = A + sp.eye(A.shape[0])
        A = np.array(A1, dtype=int)
        # print(A)
        if len(A[0]) > max_node:
            a = A[0:max_node, 0:max_node]
            # print(aa)
        else:
            a = np.zeros((max_node, max_node), dtype=int)
            # A = A + a
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    a[i][j] = A[i][j]
        a2 = a.dot(a)
        # print(a2)
        a3 = a.dot(a2)
        a4 = a.dot(a3)
        a5 = a.dot(a4)
        a6 = a.dot(a5)

        A2 = normalize_data(a2)
        A2 = np.array(A2)
        A2 = sparse.csr_matrix(A2)
        A2 = torch.FloatTensor(np.array(A2.todense()))
        # A2 = sparse_mx_to_torch_sparse_tensor(A2)
        aa2.append(A2)

        A3 = normalize_data(a3)
        A3 = np.array(A3)
        A3 = sparse.csr_matrix(A3)
        A3 = torch.FloatTensor(np.array(A3.todense()))
        aa3.append(A3)

        a = np.array(a, dtype=float)
        adj = normalize(a)
        # print(adj)
        adj = sp.csr_matrix(adj)
        adj = torch.FloatTensor(np.array(adj.todense()))
        # print(adj)
        a1.append(adj)
    # print(len(a1))

    return a1, aa2, aa3


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# 计算A^=(D~)^0.5 A~ (D~)^0.5这个公式
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_data(data):
    list = []
    for line in data:
        q = np.sum(line ** 2)
        # print(q)
        if q != 0:
            normalized_line = line / np.sqrt(q)
            list.append(normalized_line)
        else:
            list.append(line)
    return list
