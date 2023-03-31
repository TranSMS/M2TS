import argparse
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, random_split
import torch.utils.data as Data
from Model.get_A import read_batchA
from Model.get_node_embedding import get_embed
from Model.util import epoch_time
from Model.MyDataset import MySet, MySampler
from Model.gcn_encoder import GCNEncoder
# from transformer2 import Transformer2
from Model.model import Transformer
from Model.trains import train, evaluate
from Model.make_data import load_nl_data, load_code_data
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument("-b", "--batch_size", type=int, default=32, help="number of batch_size")
parser.add_argument("-e", "--epochs", type=int, default=150, help="number of epochs")
parser.add_argument("-sn", "--max_ast_node_num", type=int, default=60, help="AST graph max node number")
parser.add_argument("-s", "--src_max_len", type=int, default=300, help="maximum sequence len")
parser.add_argument("-n", "--nl_max_len", type=int, default=30, help="maximum nl len")
parser.add_argument("-dp", "--dropout", type=float, default=0.1, help="maximum sequence len")
parser.add_argument("-fd", "--nfeat_dim", type=int, default=768, help="graph hidden dimension")
parser.add_argument("-hd", "--nhid_dim", type=int, default=768, help="graph hidden dimension")
parser.add_argument("-od", "--nout_dim", type=int, default=768, help="graph hidden dimension")
parser.add_argument("-l", "--layers", type=int, default=6, help="number of layers")
parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate of adam")
parser.add_argument("-k", "--dk", type=int, default=64, help="transfrmer dimension")
parser.add_argument("-v", "--dv", type=int, default=64, help="Transfrmer dimension")
parser.add_argument("-ff", "--dff", type=int, default=2048, help="Transfrmer dimension")
parser.add_argument("-m", "--dmodel", type=int, default=512, help="Transfrmer dimension")

parser.add_argument("-tn", "--train_num", type=int, default=69708, help="training number")
args = parser.parse_args()

tgt_vocab_size, tgt_inv_vocab_dict, dec_inputs, tgt_vocab, dec_outputs = load_nl_data('../data/train/train.nl', nl_max_len=args.nl_max_len)
src_vocab_size, enc_inputs, src_vocab = load_code_data('../data/train/train.code', args.src_max_len)

# 首先将源代码解析为AST
A, A2, A3, A4, A5 = read_batchA('../data/train/train_ast.txt', args.max_ast_node_num)
X = get_embed('../data/train/train_ast.txt', args.max_ast_node_num)

A_1 = A[0:args.train_num]
A_2 = A[args.train_num:len(A)]
A2_1 = A2[0:args.train_num]
A2_2 = A2[args.train_num:len(A2)]
A3_1 = A3[0:args.train_num]
A3_2 = A3[args.train_num:len(A3)]
A4_1 = A4[0:args.train_num]
A4_2 = A4[args.train_num:len(A4)]
A5_1 = A5[0:args.train_num]
A5_2 = A5[args.train_num:len(A5)]

X_1 = X[0:args.train_num]
X_2 = X[args.train_num:len(X)]

enc_inputs = torch.LongTensor(enc_inputs)
dec_inputs = torch.LongTensor(dec_inputs)
dec_outputs = torch.LongTensor(dec_outputs)

enc_1 = enc_inputs[:args.train_num]
enc_2 = enc_inputs[args.train_num:]
dec_in_1 = dec_inputs[:args.train_num]
dec_in_2 = dec_inputs[args.train_num:]
dec_out_1 = dec_outputs[:args.train_num]
dec_out_2 = dec_outputs[args.train_num:]

train_data = MySet(A_1, X_1, A2_1, A3_1, A4_1, A5_1, enc_1, dec_in_1, dec_out_1)
evl_data = MySet(A_2, X_2, A2_2, A3_2, A4_2, A5_2, enc_2, dec_in_2, dec_out_2)

my_sampler1 = MySampler(train_data, args.batch_size)
my_sampler2 = MySampler(evl_data, args.batch_size)
evl_data_loader = DataLoader(evl_data, batch_sampler=my_sampler2)
train_data_loader = DataLoader(train_data, batch_sampler=my_sampler1)
gcn_model = GCNEncoder().to(device)
trans_model = Transformer(src_vocab_size, tgt_vocab_size, max_ast_node=args.max_ast_node_num, src_max_length=args.src_max_len,
                          nfeat=args.nfeat_dim, nhid=args.nhid_dim, nout=args.nout_dim, d_model=args.dmodel, batch_size=args.batch_size, dropout=args.dropout,
                          d_k=args.dk, d_v=args.dv, d_ff=args.dff, n_heads=args.attn_heads, n_layers=args.layers, device=device).to(device)
# trans2_model = Transformer2(src_vocab_size, tgt_vocab_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
LEARNING_RATE = args.lr
N_EPOCHS = args.epochs
tran_optimizer = optim.SGD(trans_model.parameters(), lr=LEARNING_RATE, momentum=0.99)


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


best_test_loss = float('inf')

for epoch in range(N_EPOCHS):
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
        torch.save(trans_model.state_dict(), 'save_model/trans.pt')


def beam_search(trans_model, enc_input, x, a1, a2, a3, a4, a5, start_symbol, nl_max_len):  # 变动

    enc_outputs, enc_self_attns = trans_model.encoder(enc_input)
    dec_input = torch.zeros(1, nl_max_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, nl_max_len):
        dec_input[0][i] = next_symbol
        ast_outputs, ast_embed = trans_model.gcn_encoder(x, a1, a2, a3, a4, a5)
        dec_outputs, _, _, _ = trans_model.decoder1(dec_input, enc_input, enc_outputs, ast_outputs, ast_embed)  # 变动
        projected = trans_model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input


def predict():
    trans_model.load_state_dict(torch.load('save_model/trans.pt'))
    trans_model.eval()
    a, x, a2, a3, a4, a5, inputs, _, _ = next(iter(evl_data_loader))
    for j in range(len(inputs)):
        a, x, a2, a3, a4, a5 = a.to(device), x.to(device), a2.to(device), a3.to(device), a4.to(device), a5.to(device)
        greedy_dec_input = beam_search(trans_model, inputs[j].view(1, -1).to(device), x, a, a2, a3, a4, a5, start_symbol=tgt_vocab['SOS'], nl_max_len=args.nl_max_len)  # 变动
        pred, _, _, _, _ = trans_model(inputs[j].view(1, -1).to(device), greedy_dec_input, x, a, a2, a3, a4, a5)  # 变动
        pred = pred.data.max(1, keepdim=True)[1]

        summary = [tgt_inv_vocab_dict[n.item()] for n in pred.squeeze()]
        print(summary)

if __name__ == '__main__':
    predict()
