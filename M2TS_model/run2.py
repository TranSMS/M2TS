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
from trans_model2 import Transformer
from train_eval2 import train, evaluate
from make_data import load_nl_data, load_code_data
import torch.optim as optim
import argparse
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

parser = argparse.ArgumentParser()
parser.add_argument('--epoches', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate.')
parser.add_argument('--nl_length', type=int, default=30,
                    help='NL-MAX-Length.')
parser.add_argument('--AST_Node', type=int, default=30,
                    help='Number of AST Nodes.')
parser.add_argument('--Train_data', type=int, default=62738,
                    help='Number of training data.')
parser.add_argument('--code_length', type=int, default=300,
                    help='code-MAX-Length.')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Number of the batch.')
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tgt_vocab_size, tgt_inv_vocab_dict, dec_inputs, tgt_vocab, dec_outputs = load_nl_data('data/train_nl1.txt', args.nl_length)
src_vocab_size, enc_inputs, src_vocab, src_inv_vocab_size = load_code_data('data/train_code1.txt', args.code_length)
print(src_vocab)
# exit()
A, A2, A3, A4, A5 = read_batchA('data/train_ast1.txt', args.AST_node)
X = get_embed('data/train_ast1.txt', args.AST_node)

A_1 = A[0:args.Train_data]
A_2 = A[args.Train_data:len(A)]
A2_1 = A2[0:args.Train_data]
A2_2 = A2[args.Train_data:len(A2)]
A3_1 = A3[0:args.Train_data]
A3_2 = A3[args.Train_data:len(A3)]

X_1 = X[0:args.Train_data]
X_2 = X[args.Train_data:len(X)]

enc_inputs = torch.LongTensor(enc_inputs)
dec_inputs = torch.LongTensor(dec_inputs)
dec_outputs = torch.LongTensor(dec_outputs)

enc_1 = enc_inputs[:args.Train_data]
enc_2 = enc_inputs[args.Train_data:]
dec_in_1 = dec_inputs[:args.Train_data]
dec_in_2 = dec_inputs[args.Train_data:]
dec_out_1 = dec_outputs[:args.Train_data]
dec_out_2 = dec_outputs[args.Train_data:]

# dataset = MySet(A, X, A2, A3, A4, A5, enc_inputs, dec_inputs, dec_outputs)
train_data = MySet(A_1, X_1, A2_1, A3_1, enc_1, dec_in_1, dec_out_1)
evl_data = MySet(A_2, X_2, A2_2, A3_2, enc_2, dec_in_2, dec_out_2)
# train_data, evl_data = random_split(dataset, [1040, 260])
# exit()
my_sampler1 = MySampler(train_data, args.batch_size)
my_sampler2 = MySampler(evl_data, args.batch_size)
evl_data_loader = DataLoader(evl_data, batch_sampler=my_sampler2)
train_data_loader = DataLoader(train_data, batch_sampler=my_sampler1)
gcn_model = GCNEncoder().to(device)
trans_model = Transformer(src_vocab_size, tgt_vocab_size).to(device)
# trans2_model = Transformer2(src_vocab_size, tgt_vocab_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
gcn_optimizer = optim.SGD(gcn_model.parameters(), lr=args.lr, momentum=0.99)
tran_optimizer = optim.SGD(trans_model.parameters(), lr=args.lr, momentum=0.99)

best_test_loss = float('inf')

for epoch in range(args.epoches):
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
        torch.save(gcn_model.state_dict(), 'save_model/gcn_model2.pt')
        torch.save(trans_model.state_dict(), 'save_model/trans_loss2.pt')
        # torch.save(trans2_model.state_dict(), 'save_model/multi_loss2.pt')

# exit()


def greedy_decoder(trans_model, enc_input, ast_outputs, start_symbol):  # 变动

    enc_outputs, enc_self_attns = trans_model.encoder(enc_input)
    dec_input = torch.zeros(1, args.nl_length).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, args.nl_length):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _, _ = trans_model.decoder2(dec_input, enc_input, enc_outputs, ast_outputs)  # 变动
        projected = trans_model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input


def nltk_sentence_bleu(hypotheses, references, order=4):
    refs = []
    count = 0
    total_score = 0.0
    cc = SmoothingFunction()
    for hyp, ref in zip(hypotheses, references):
        hyp = hyp.split()
        ref = ref.split()
        refs.append([ref])
        if len(hyp) < order:
            continue
        else:
            score = nltk.translate.bleu([ref], hyp, smoothing_function=cc.method4)
            total_score += score
            count += 1
    avg_score = total_score / count
    hpy2 = []
    for i in hypotheses:
        # print(i)
        i = i.split()
        hpy2.append(i)
    # print(refs)
    # corpus_bleu = nltk.translate.bleu_score.corpus_bleu(refs, hpy2, weights=(0.25, 0.25, 0.25, 0.25))
    # print('corpus_bleu: %.4f sentence_bleu: %.4f' % (corpus_bleu, avg_score))
    return avg_score


def meteor_score1(hypothesis, reference):
    count = 0
    total_score = 0.0
    for i in range(len(hypothesis)):
        score = round(meteor_score([reference[i]], hypothesis[i]), 4)
        # print(score)
        # exit()
        total_score += score
        count += 1
    avg_score = total_score/count
    # print('METEOR_score: %.4f' % avg_score)
    return avg_score


def predict():
    # Model = Transformer().to(device)
    gcn_model.load_state_dict(torch.load('save_model/gcn_model2.pt'))
    trans_model.load_state_dict(torch.load('save_model/trans_loss2.pt'))
    # trans2_model.load_state_dict(torch.load('save_model/multi_loss2.pt'))
    gcn_model.eval()
    trans_model.eval()
    # trans2_model.eval()
    # torch.no_grad()
    # a, x, a2, a3, a4, a5, inputs, _, _ = next(iter(evl_data_loader))
    for a1, x1, a2_1, a3_1, a4_1, a5_1, input1, _, _ in evl_data_loader:
        print(a1)
        print(a1.shape)
        q = []
        for j in range(len(input1)):
            # a1, x1, a2_1, a3_1, a4_1, a5_1 = a1.to(device), x1.to(device), a2_1.to(device), a3_1.to(device), a4_1.to(device), a5_1.to(device)
            ast_outputs, ast_embed = gcn_model(x1[j].unsqueeze(0), a1[j].unsqueeze(0), a2_1[j].unsqueeze(0), a3_1[j].unsqueeze(0), a4_1[j].unsqueeze(0), a5_1[j].unsqueeze(0))  # 变动
            # print(ast_outputs.shape)
            # exit()
            greedy_dec_input = greedy_decoder(trans_model, input1[j].view(1, -1).to(device), ast_outputs, start_symbol=tgt_vocab['SOS'])  # 变动
            pred, _, _, _, _ = trans_model(input1[j].view(1, -1).to(device), greedy_dec_input, ast_outputs)  # 变动
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
        with open('data/ref1.txt', 'r', encoding='utf-8') as f:  # ref1
            lines = f.readlines()

            for line in lines:
                line = line.strip('\n')
            # print(line)
                ref.append(line)
    # print(ref)

        avg_score = nltk_sentence_bleu(pred1, ref)
        meteor = meteor_score1(pred1, ref)
        print('S_BLEU: %.4f' % avg_score)
        # print('C-BLEU: %.4f' % corup_BLEU)
        print('METEOR: %.4f' % meteor)
        rouge = Rouge()
        rough_score = rouge.get_scores(pred1, ref, avg=True)
        print(' ROUGE: ', rough_score)

    # print(x1)
    #     print([tgt_inv_vocab_dict[n.item()] for n in pred.squeeze()])

if __name__ == '__main__':
    predict()
