from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

batch_size = 8
nfeat= 768
nhid = 768
nout = 768
dropout = 0.2
d_model = 512
# d_model = 512
d_ff = 1024  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 12  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # torch.FloatTensor使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化
        self.weight = Parameter(torch.FloatTensor(batch_size, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 初始化权重
    def reset_parameters(self):
        # size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数  size（1）为行
        stdv = 1. / math.sqrt(self.weight.size(1))
        # uniform() 方法将随机生成下一个实数，它在 [x, y] 范围内
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    '''
    前馈运算 即计算A~ X W(0)
    input X与权重W相乘，然后adj矩阵与他们的积稀疏乘
    直接输入与权重之间进行torch.mm操作，得到support，即XW
    support与adj进行torch.spmm操作，得到output，即AXW选择是否加bias
    '''
    # 这里的input指的是节点的word embedding的维度
    def forward(self, input, adj):
        outputs = torch.zeros_like(input)
        for i in range(input.size(0)):
            support = torch.mm(input[i], self.weight[i])
            # spmm的意思是考虑加不加偏置
            output = torch.mm(adj[i], support)
            if self.bias is not None:
                output + self.bias
            outputs[i] = output
        # print(outputs)
        return outputs

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# 定义GCN类
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout
        # self.ffn = nn.Linear(nout, d_model)

    # 输入分别是特征和邻接矩阵
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        outputs = F.relu(self.gc2(x, adj))
        # outputs = self.ffn(outputs)
        return outputs


class AST_Model(nn.Module):
    def __init__(self):
        super(AST_Model, self).__init__()
        self.gcn1 = GCN()
        self.gcn2 = GCN()
        self.gcn3 = GCN()
        self.gcn4 = GCN()
        self.gcn5 = GCN()
        self.gcn6 = GCN()
        self.gcn7 = GCN()
        self.gcn8 = GCN()
        self.gcn9 = GCN()
        self.gcn10 = GCN()
        self.ffn = nn.Linear(nout, d_model)

    def forward(self, x, adj, A2, A3, A4, A5, A6, A7, A8, A9, A10):
        output1 = self.gcn1(x, adj)
        output2 = self.gcn2(output1, A2)
        output3 = self.gcn3(output2, A3)
        output4 = self.gcn4(output3, A4)
        output5 = self.gcn5(output4, A5)
        output6 = self.gcn6(output5, A6)
        output7 = self.gcn7(output6, A7)
        output8 = self.gcn8(output7, A8)
        output9 = self.gcn9(output8, A9)
        output10 = self.gcn10(output9, A10)
        gcn_output = output1
        gcn_output = self.ffn(gcn_output)
        # print(gcn_output)
        # print(gcn_output.shape)

        return gcn_output


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                    2)   # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
        #                                           1)   # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        # output = self.fc(context)  # [batch_size, len_q, d_model]
        # 残差和层归一化
        return context, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, ast_outputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(ast_outputs, ast_outputs, ast_outputs)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class GCNEncoder(nn.Module):
    def __init__(self):
        super(GCNEncoder, self).__init__()
        # self.src_vocab_size = src_vocab_size
        # self.src_emb = nn.Embedding(self.src_vocab_size, d_model)
        # self.pos_emb = PositionalEncoding(d_model)
        # 设定encoder的个数
        self.ast_output = AST_Model()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, x, a, a2, a3, a4, a5, a6, a7, a8, a9, a10):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        ast_embed = self.ast_output(x, a, a2, a3, a4, a5, a6, a7, a8, a9, a10)  # 变动
        # enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        # enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        # enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        ast_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            ast_outputs, enc_self_attn = layer(ast_embed)  # 变动
            ast_self_attns.append(enc_self_attn)
        return ast_outputs, ast_embed  # 变动