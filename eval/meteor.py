def meteor_score(hypothesis, reference):
    count = 0
    total_score = 0.0
    for i in range(len(hypothesis)):
        score = round(meteor_score([reference[i]], hypothesis[i]), 4)

        total_score += score
        count += 1
    avg_score = total_score/count
    # print('METEOR_score: %.4f' % avg_score)
    return avg_score