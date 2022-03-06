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

# print(' ')
# s_bleu = nltk_sentence_bleu(cad, ref)
# meteor = meteor_score1(cad, ref)
# print(s_bleu)
# print(meteor)
# rouge = Rouge()
# rough_score = rouge.get_scores(cad, ref, avg=True)
# print(rough_score)