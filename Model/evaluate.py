from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import nltk


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
    S_BLEU = total_score / count
    # hpy2 = []
    # for i in hypotheses:
    #     # print(i)
    #     i = i.split()
    #     hpy2.append(i)

    return S_BLEU


def meteor_score(hypothesis, reference):
    count = 0
    total_score = 0.0
    for i in range(len(hypothesis)):
        score = round(meteor_score([reference[i]], hypothesis[i]), 4)
        # print(score)
        # exit()
        total_score += score
        count += 1
    METEOR = total_score/count
    # print('METEOR_score: %.4f' % avg_score)
    return METEOR
    
# ref.txt为测试集真值
ref = []
with open('../Model/data/ref.txt', 'r', encoding='utf-8') as f:  # ref1
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        # print(line)
        ref.append(line)
hyp = []
with open('../Model/data/hyp.txt', 'r', encoding='utf-8') as ff:  # ref1
    lines = ff.readlines()
    for line in lines:
        line = line.strip('\n')
        # print(line)
        hyp.append(line)
avg_score = nltk_sentence_bleu(hyp, ref)
print('S_BLEU: %.4f' % avg_score)
meteor = meteor_score(hyp, ref)
print('METEOR: %.4f' % meteor)
rouge = Rouge()
rough_score = rouge.get_scores(pred1, ref, avg=True)
print(' ROUGE: ', rough_score)



