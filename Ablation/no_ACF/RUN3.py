import torch
import time
import math
from visdom import Visdom
import nltk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, random_split
import torch.utils.data as Data
from model.get_A import read_batchA
from model.get_embed import get_embed
from model.util import epoch_time
from model.MyData import MySet, MySampler
from model.GCN_encoder import GCNEncoder
# from transformer2 import Transformer2
from ablation_model.no_ACF.Model2 import Transformer
from ablation_model.no_ACF.train_test2 import train, evaluate
from model.make_data import load_nl_data, load_code_data
import torch.optim as optim
from model.metrics import nltk_sentence_bleu, meteor_score
from rouge import Rouge
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
batch_size = 8
epoches = 180
nl_max_len = 16
# seq_max_len = 111
train_num = 68  # 960
max_ast_node = 23  # 60
src_max_length = 65  # 120
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tgt_vocab_size, tgt_inv_vocab_dict, dec_inputs, tgt_vocab, dec_outputs = load_nl_data('../../datasets/Java_DeepCom/training/summary.txt', nl_max_len)
src_vocab_size, enc_inputs, src_vocab = load_code_data('../../datasets/Java_DeepCom/training/code.txt', src_max_length)
# print(src_vocab)
# print(tgt_vocab)
# print(tgt_vocab_size)
# exit()
A, A2, A3, A4, A5 = read_batchA('../../datasets/Java_DeepCom/training/AST.txt', max_ast_node)
X = get_embed('../../datasets/Java_DeepCom/training/AST.txt', max_ast_node)

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
train_data = MySet(A_1, X_1, A2_1, A3_1, A4_1, A5_1, enc_1, dec_in_1, dec_out_1)
evl_data = MySet(A_2, X_2, A2_2, A3_2, A4_2, A5_2, enc_2, dec_in_2, dec_out_2)
# train_data, evl_data = random_split(dataset, [1040, 260])
# exit()
my_sampler1 = MySampler(train_data, batch_size)
my_sampler2 = MySampler(evl_data, batch_size)
evl_data_loader = DataLoader(evl_data, batch_sampler=my_sampler2)
train_data_loader = DataLoader(train_data, batch_sampler=my_sampler1)

# trans_loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size=batch_size, shuffle=True)
# gcn_model = GCNEncoder().to(device)
trans_model = Transformer(src_vocab_size, tgt_vocab_size).to(device)
# trans2_model = Transformer2(src_vocab_size, tgt_vocab_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
# gcn_optimizer = optim.SGD(gcn_model.parameters(), lr=0.0001, momentum=0.99)
tran_optimizer = optim.SGD(trans_model.parameters(), lr=0.0001, momentum=0.99)
# exit()


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    # trainable_num = sum(p.numel for p in model.parameters() if p.requires_grad)
    return total_num

# 参数数量
# Total1 = get_parameter_number(gcn_model)
# Total2 = get_parameter_number(trans_model)
# print('num_1:', Total1)
# print('num_2:', Total2)
# exit()

# viz = Visdom()
# viz.line([0.], [0.], win='train_loss', opts=dict(title='train_loss'))
# viz.line([0.], [0.], win='val_loss', opts=dict(title='val_loss'))


best_test_loss = float('inf')
for epoch in range(epoches):
    start_time = time.time()
    train_loss = train(tran_optimizer, train_data_loader, trans_model, criterion, device)
    eval_loss, perplexity = evaluate(evl_data_loader, trans_model, criterion, device)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print('Epoch:', '%04d' % (epoch + 1),  f'Time: {epoch_mins}m {epoch_secs}s')
    print('\ttrain loss: ', '{:.4f}'.format(train_loss))
    print('\t eval_loss: ', '{:.4f}'.format(eval_loss))
    print('\tperplexity: ', '{:.4f}'.format(perplexity))
    if eval_loss < best_test_loss:
        best_test_loss = eval_loss
        # torch.save(gcn_model.state_dict(), 'save_model/gcn_model.pt')
        torch.save(trans_model.state_dict(), 'trans3.pt')
        # torch.save(trans2_model.state_dict(), 'save_model/multi_loss2.pt')

    # viz.line([train_loss], [epoch], win='train_loss', update='append')
    # viz.line([eval_loss], [epoch], win='val_loss', update='append')


exit()
def beam_search(trans_model, enc_input, ast_outputs, ast_embed, start_symbol):  # 变动

    enc_outputs, enc_self_attns = trans_model.encoder(enc_input)
    dec_input = torch.zeros(1, nl_max_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, nl_max_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _, _ = trans_model.decoder1(dec_input, enc_input, enc_outputs, ast_outputs, ast_embed)  # 变动
        projected = trans_model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input


def predict():
    # Model = Transformer().to(device)
    gcn_model.load_state_dict(torch.load('save_model/gcn_model.pt'))
    trans_model.load_state_dict(torch.load('save_model/trans_loss1.pt'))
    # trans2_model.load_state_dict(torch.load('save_model/multi_loss2.pt'))
    gcn_model.eval()
    trans_model.eval()
    # trans2_model.eval()
    # torch.no_grad()
    # for a1, x1, a2_1, a3_1, a4_1, a5_1, input1, _, _ in evl_data_loader:
    #     print(a1)
    a, x, a2, a3, a4, a5, inputs, _, _ = next(iter(evl_data_loader))

    q = []
    for j in range(len(inputs)):
        a, x, a2, a3, a4, a5 = a.to(device), x.to(device), a2.to(device), a3.to(device), a4.to(device), a5.to(device)
        ast_outputs, ast_embed = gcn_model(x[j].unsqueeze(0), a[j].unsqueeze(0), a2[j].unsqueeze(0), a3[j].unsqueeze(0), a4[j].unsqueeze(0), a5[j].unsqueeze(0), a6[j].unsqueeze(0), a7[j].unsqueeze(0), a8[j].unsqueeze(0), a9[j].unsqueeze(0), a10[j].unsqueeze(0))  # 变动
        # print(ast_outputs.shape)
        # exit()
        greedy_dec_input = beam_search(trans_model, inputs[j].view(1, -1).to(device), ast_outputs, ast_embed, start_symbol=tgt_vocab['SOS'])  # 变动
        pred, _, _, _, _ = trans_model(inputs[j].view(1, -1).to(device), greedy_dec_input, ast_outputs, ast_embed)  # 变动
        pred = pred.data.max(1, keepdim=True)[1]
        for i in range(len(pred)):
            if i > 0 and pred[i] == 3:
                pred = pred[0:i+1]
                break
            else:
                continue
        x1 = [tgt_inv_vocab_dict[n.item()] for n in pred.squeeze()]
        q.append(x1)
    # print(q)
    pred1 = []
    for k in q:
        s = " ".join(k)
        pred1.append(s)
    # print(pred1)
    with open('data/hyp.txt', 'w', encoding='utf-8') as ff:
        for z in pred1:
            ff.writelines(z + '\n')
    ref = []
    with open('data/ref2.txt', 'r', encoding='utf-8') as f:  # ref1
        lines = f.readlines()

        for line in lines:
            line = line.strip('\n')
            # print(line)
            ref.append(line)
    # print(ref)
    avg_score = nltk_sentence_bleu(pred1, ref)
    meteor = meteor_score(pred1, ref)
    print('S_BLEU: %.4f' % avg_score)
    # print('C-BLEU: %.4f' % corup_BLEU)
    print('METEOR: %.4f' % meteor)
    rouge = Rouge()
    rough_score = rouge.get_scores(pred1, ref, avg=True)
    print(' ROUGE: ', rough_score)


if __name__ == '__main__':
    predict()
