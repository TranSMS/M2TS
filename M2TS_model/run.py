import torch
import time
import math
from visdom import Visdom
# from util import epoch_time
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import nltk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, random_split
import torch.utils.data as Data
from get_A import read_batchA
from get_embed import get_embed
from util import epoch_time
from MySet import MySet, MySampler
from gcn_model import AST_Model, GCNEncoder
# from transformer2 import Transformer2
from trans_model import Transformer
from train_eval import train, evaluate
from make_data import load_nl_data, load_code_data
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
batch_size = 8
epoches = 200
nl_max_len = 16
# seq_max_len = 111
train_num = 68  # 960
max_ast_node = 23  # 60
src_max_length = 65  # 120
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tgt_vocab_size, tgt_inv_vocab_dict, dec_inputs, tgt_vocab, dec_outputs = load_nl_data('data/train_nl1.txt', nl_max_len)
src_vocab_size, enc_inputs, src_vocab = load_code_data('data/train_code1.txt', src_max_length)
# print(src_vocab)
# print(tgt_vocab)
# print(tgt_vocab_size)
# exit()
A, A2, A3, A4, A5, A6, A7, A8, A9, A10 = read_batchA('data/train_ast1.txt', max_ast_node)
X = get_embed('data/train_ast1.txt', max_ast_node)

A_1 = A[0:train_num]
A_2 = A[train_num:len(A)]
# print(A_2)
A2_1 = A2[0:train_num]
A2_2 = A2[train_num:len(A2)]
A3_1 = A3[0:train_num]
A3_2 = A3[train_num:len(A3)]
A4_1 = A4[0:train_num]
A4_2 = A4[train_num:len(A4)]
A5_1 = A5[0:train_num]
A5_2 = A5[train_num:len(A5)]
A6_1 = A6[0:train_num]
A6_2 = A6[train_num:len(A6)]
A7_1 = A7[0:train_num]
A7_2 = A7[train_num:len(A7)]
A8_1 = A8[0:train_num]
A8_2 = A8[train_num:len(A8)]
A9_1 = A9[0:train_num]
A9_2 = A9[train_num:len(A9)]
A10_1 = A10[0:train_num]
A10_2 = A10[train_num:len(A10)]

X_1 = X[0:train_num]
X_2 = X[train_num:len(X)]

enc_inputs = torch.LongTensor(enc_inputs)
dec_inputs = torch.LongTensor(dec_inputs)
dec_outputs = torch.LongTensor(dec_outputs)

enc_1 = enc_inputs[:train_num]
enc_2 = enc_inputs[train_num:]
dec_in_1 = dec_inputs[:train_num]
dec_in_2 = dec_inputs[train_num:]
dec_out_1 = dec_outputs[:train_num]
dec_out_2 = dec_outputs[train_num:]

# exit()
# dataset = MySet(A, X, A2, A3, A4, A5, enc_inputs, dec_inputs, dec_outputs)
train_data = MySet(A_1, X_1, A2_1, A3_1, A4_1, A5_1, A6_1, A7_1, A8_1, A9_1, A10_1, enc_1, dec_in_1, dec_out_1)
evl_data = MySet(A_2, X_2, A2_2, A3_2, A4_2, A5_2, A6_2, A7_2, A8_2, A9_2, A10_2, enc_2, dec_in_2, dec_out_2)
# train_data, evl_data = random_split(dataset, [1040, 260])
# exit()
my_sampler1 = MySampler(train_data, batch_size)
my_sampler2 = MySampler(evl_data, batch_size)
evl_data_loader = DataLoader(evl_data, batch_sampler=my_sampler2)
train_data_loader = DataLoader(train_data, batch_sampler=my_sampler1)

# trans_loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size=batch_size, shuffle=True)
gcn_model = GCNEncoder().to(device)
trans_model = Transformer(src_vocab_size, tgt_vocab_size, max_ast_node, src_max_length).to(device)
# trans2_model = Transformer2(src_vocab_size, tgt_vocab_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
gcn_optimizer = optim.SGD(gcn_model.parameters(), lr=0.0001, momentum=0.99)
tran_optimizer = optim.SGD(trans_model.parameters(), lr=0.0001, momentum=0.99)
# exit()
best_test_loss = float('inf')
viz = Visdom()
viz.line([0.], [0.], win='train_loss', opts=dict(title='train_loss'))
viz.line([0.], [0.], win='val_loss', opts=dict(title='val_loss'))


for epoch in range(epoches):
    start_time = time.time()
    train_loss = train(gcn_optimizer, tran_optimizer, train_data_loader, gcn_model, trans_model, criterion, device)
    eval_loss, perplexity = evaluate(evl_data_loader, gcn_model, trans_model, criterion, device)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print('Epoch:', '%04d' % (epoch + 1),  f'Time: {epoch_mins}m {epoch_secs}s')
    print('\ttrain loss: ', '{:.4f}'.format(train_loss))
    print('\t eval_loss: ', '{:.4f}'.format(eval_loss))
    print('\tperplexity: ', '{:.4f}'.format(perplexity))
    if eval_loss < best_test_loss:
        best_test_loss = eval_loss
        torch.save(gcn_model.state_dict(), 'save_model/gcn_model.pt')
        torch.save(trans_model.state_dict(), 'save_model/trans_loss1.pt')
        # torch.save(trans2_model.state_dict(), 'save_model/multi_loss2.pt')

    viz.line([train_loss], [epoch], win='train_loss', update='append')
    viz.line([eval_loss], [epoch], win='val_loss', update='append')
