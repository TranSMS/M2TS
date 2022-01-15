import torch.utils.data as Data
from torch.utils.data import Dataset, Sampler


class MySet(Dataset):

    # 读取数据
    def __init__(self, adj, X, a2, a3, enc_inputs, dec_inputs, dec_outputs):
        self.adj = adj
        self.X = X
        self.a2 = a2
        self.a3 = a3

        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
    # 根据索引返回数据

    def __getitem__(self, idx):
        return self.adj[idx], self.X[idx], self.a2[idx], self.a3[idx], self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

    # 返回数据集总长度
    def __len__(self):
        return len(self.adj)


class MySampler(Sampler):
    def __init__(self, dataset, batchsize):
        super(Sampler, self).__init__()
        self.dataset = dataset
        self.batch_size = batchsize		# 每一批数据量
        self.indices = range(len(dataset))	 # 生成数据集的索引
        self.count = int(len(dataset) / self.batch_size)  # 一共有多少批

    def __iter__(self):
        for i in range(self.count):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return self.count
