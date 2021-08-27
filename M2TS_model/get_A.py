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
    aa4 = []
    aa5 = []
    aa6 = []
    aa7 = []
    aa8 = []
    aa9 = []
    aa10 = []
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
        a7 = a.dot(a6)
        a8 = a.dot(a7)
        a9 = a.dot(a8)
        a10 = a.dot(a9)

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

        A4 = normalize_data(a4)
        A4 = np.array(A4)
        A4 = sparse.csr_matrix(A4)
        A4 = torch.FloatTensor(np.array(A4.todense()))
        aa4.append(A4)

        A5 = normalize_data(a5)
        A5 = np.array(A5)
        A5 = sparse.csr_matrix(A5)
        A5 = torch.FloatTensor(np.array(A5.todense()))
        aa5.append(A5)

        A6 = normalize_data(a6)
        A6 = np.array(A6)
        A6 = sparse.csr_matrix(A6)
        A6 = torch.FloatTensor(np.array(A6.todense()))
        aa6.append(A6)

        A7 = normalize_data(a7)
        A7 = np.array(A7)
        A7 = sparse.csr_matrix(A7)
        A7 = torch.FloatTensor(np.array(A7.todense()))
        aa7.append(A7)

        A8 = normalize_data(a8)
        A8 = np.array(A8)
        A8 = sparse.csr_matrix(A8)
        A8 = torch.FloatTensor(np.array(A8.todense()))
        aa8.append(A8)

        A9 = normalize_data(a9)
        A9 = np.array(A9)
        A9 = sparse.csr_matrix(A9)
        A9 = torch.FloatTensor(np.array(A9.todense()))
        aa9.append(A9)

        A10 = normalize_data(a10)
        A10 = np.array(A10)
        A10 = sparse.csr_matrix(A10)
        A10 = torch.FloatTensor(np.array(A10.todense()))
        aa10.append(A10)

        a = np.array(a, dtype=float)
        adj = normalize(a)
        # print(adj)
        adj = sp.csr_matrix(adj)
        adj = torch.FloatTensor(np.array(adj.todense()))
        # print(adj)
        a1.append(adj)
    # print(len(a1))

    return a1, aa2, aa3, aa4, aa5, aa6, aa7, aa8, aa9, aa10


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
