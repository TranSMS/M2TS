import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from model.make_data import load_nl_data, load_code_data
import nltk
from collections import Counter
from ablation_model.no_AST.MyData import MySet, MySampler
from torch.utils.data import DataLoader, Sampler, random_split, Dataset
from visdom import Visdom
import time
from model.util import epoch_time
from ablation_model.no_AST.Transformer import Transformer
from ablation_model.no_AST.train_eval import train, evaluate
from model.metrics import meteor_score, nltk_sentence_bleu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src_len = 65  # enc_input max sequence length
tgt_len = 16  # dec_input(=dec_output) max sequence length
batch_size = 8
epoches = 180
train_num = 68

code_total_words, code_inputs, code_word_dict = load_code_data('../../datasets/Java_DeepCom/training/code.txt', src_len)
nl_total_words, nl_inv_word_dict, nl_inputs, nl_word_dict, nl_outputs = load_nl_data('../../datasets/Java_DeepCom/training/summary.txt', tgt_len)
# print(nl_word_dict)
# exit()
code_inputs = torch.LongTensor(code_inputs)
nl_inputs = torch.LongTensor(nl_inputs)
nl_outputs = torch.LongTensor(nl_outputs)

code_1 = code_inputs[:train_num]
code_2 = code_inputs[train_num:]
nl_in_1 = nl_inputs[:train_num]
nl_in_2 = nl_inputs[train_num:]
nl_out_1 = nl_outputs[:train_num]
nl_out_2 = nl_outputs[train_num:]
# print(code_2)
# print(code_2.shape)
# exit()
dataset = MySet(code_inputs, nl_inputs, nl_outputs)
# print(dataset)
# train_data, test_data = random_split(dataset, [8430, 2107])
train_data = MySet(code_1, nl_in_1, nl_out_1)
test_data = MySet(code_2, nl_in_2, nl_out_2)

my_sampler1 = MySampler(train_data, batch_size)
my_sampler2 = MySampler(test_data, batch_size)
test_data_loader = DataLoader(test_data, batch_sampler=my_sampler2)
train_data_loader = DataLoader(train_data, batch_sampler=my_sampler1)
# inputs1, _, _ = next(iter(test_data_loader))
# print(inputs1)
# print(len(inputs1))
# inputs2 = code_2

# print(len(inputs2))
# exit()
model = Transformer(code_total_words, nl_total_words).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.99)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    # trainable_num = sum(p.numel for p in model.parameters() if p.requires_grad)
    return total_num


# Total = get_parameter_number(model)
# print('Total_num:', Total)
# exit()

# viz = Visdom()
# viz.line([0.], [0.], win='train_loss', opts=dict(title='train_loss'))
# viz.line([0.], [0.], win='val_loss', opts=dict(title='val_loss'))

best_test_loss = float('inf')
for epoch in range(epoches):

    start_time = time.time()
    train_loss = train(optimizer, train_data_loader, model, criterion, device)
    test_loss = evaluate(test_data_loader, model, criterion, device)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print('Epoch:', '%04d' % (epoch + 1), f'Time: {epoch_mins}m {epoch_secs}s')
    print('train_loss =', '{:.4f}'.format(train_loss))
    print('test_loss =', '{:.4f}'.format(test_loss))
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), 'save_model/transformer.pt')
    # viz.line([train_loss], [epoch], win='train_loss', update='append')
    # viz.line([test_loss], [epoch], win='val_loss', update='append')
# exit()


def greedy_decoder(model, enc_input, start_symbol):

    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input


def predict():
    # Model = Transformer().to(device)
    model.load_state_dict(torch.load('save_model/transformer.pt'))
    model.eval()
    # torch.no_grad()
    #inputs, _, _ = next(iter(test_data_loader))
    inputs = code_2
    # print(len(inputs))
    # print(inputs)
    q = []
    for j in range(len(inputs)):
        greedy_dec_input = greedy_decoder(model, inputs[j].view(1, -1).to(device), start_symbol=nl_word_dict['SOS'])
        pred, _, _, _ = model(inputs[j].view(1, -1).to(device), greedy_dec_input)
        predict = pred.data.max(1, keepdim=True)[1]
        for k in range(len(predict)):
            if k > 0 and predict[k] == 3:
                predict = predict[0:k + 1]
                break
            else:
                continue

            # print(j[0])
        # j = sentences[i]
        x1 = [nl_inv_word_dict[n.item()] for n in predict.squeeze()]
        q.append(x1)
    pred1 = []
    for k in q:
        s = " ".join(k)
        pred1.append(s)
    # print(pred1)
    with open('data/hyp1.txt', 'w', encoding='utf-8') as ff:
        for z in pred1:
            ff.writelines(z + '\n')

    with open('data/ref.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        l = []
        for line in lines:
            line = line.strip('\n')
            # print(line)
            l.append(line)
        # print(l)
    S_BLEU = nltk_sentence_bleu(pred1, l)
    meteor = meteor_score(pred1, l)
    print(' S_BLEU: %.4f' % S_BLEU)
    print(' METEOR: %.4f' % meteor)
    rouge = Rouge()
    rough_score = rouge.get_scores(pred1, l, avg=True)
    print('ROUGE_L: ', rough_score)
    # print('S-BLEU: %.4f' % S_BLEU)
    # print(inputs[j], '->', [nl_inv_word_dict[n.item()] for n in pred.squeeze()])


if __name__ == '__main__':
    predict()

# exit()

