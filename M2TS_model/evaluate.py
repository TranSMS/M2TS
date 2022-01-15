from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import nltk
import torch
import argparse
from run import gcn_model, trans_model, evl_data_loader
from make_data import load_nl_data
from MySet import MySet, MySampler

parser = argparse.ArgumentParser()
parser.add_argument('--nl_length', type=int, default=30,
                    help='NL-MAX-Length.')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tgt_vocab_size, tgt_inv_vocab_dict, dec_inputs, tgt_vocab, dec_outputs = load_nl_data('java_nl.txt', args.nl_length)


def beam_decoder(trans_model, enc_input, ast_outputs, start_symbol):  # 变动

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
            greedy_dec_input = beam_decoder(trans_model, input1[j].view(1, -1).to(device), ast_outputs, start_symbol=tgt_vocab['SOS'])  # 变动
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