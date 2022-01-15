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
