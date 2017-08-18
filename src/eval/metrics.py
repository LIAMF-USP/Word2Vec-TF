import pickle
import heapq
import sys
import numpy as np


def analogy(word1, word2, word3, word2index, embeddings):
    """
    Function to calculate a list of the top 10 analogues given the words
    'word1', 'word2', 'word3'.

    :type word1:str
    :type word2:str
    :type word3:str
    :type word2index: dict
    :type embeddings: np array
    :rtype result: list
    """
    assert word1 in word2index, "'{}' not in the vocab".format(word1)
    assert word2 in word2index, "'{}' not in the vocab".format(word2)
    assert word3 in word2index, "'{}' not in the vocab".format(word3)
    index2word = {v: k for k, v in word2index.items()}
    index1 = word2index[word1]
    index2 = word2index[word2]
    index3 = word2index[word3]
    wordvector1 = embeddings[index1]
    wordvector2 = embeddings[index2]
    wordvector3 = embeddings[index3]
    result_vector = embeddings.dot(wordvector2) - embeddings.dot(wordvector1) + embeddings.dot(wordvector3)

    all_results = [(v, index)
                   for index, v in enumerate(result_vector)
                   if (index != index1 and
                   index != index2 and
                   index != index3)]

    heapq._heapify_max(all_results)
    results = []
    top_results = min(len(all_results), 10)
    for _ in range(top_results):
        _, index = heapq._heappop_max(all_results)
        results.append(index2word[index])
    return results


def score(word2index,
          embeddings,
          eval_path,
          verbose=True,
          raw=False):
    """
    Function to calculate the score of the embeddings given one
    txt file of analogies "eval_path". A valid line is a line of the txt
    such that every word is in the vocabulary of the embeddings. For each
    valid line we calculate the top 10 closest words that fit the analogy for
    the first, the second and the third words of the valid line. The score of
    this line will be the position of the fourth word in this list (0 if it is
    not in the list). Since the txt can have different categories this
    function also returns a list 'results' with the different scores
    per category.

    :type word2index: dict
    :type embeddings: np array
    :type eval_path:str
    :rtype final_score: float
    :rtype results: list
    """
    old_score = 0
    old_total = 0
    old_cat = None
    valid_tests = 0
    total_lines = 0
    all_cat_scores = []
    all_cat_totals = []
    all_cat = []
    with open(eval_path) as inputfile:
        for line in inputfile:
            total_lines += 1
            list_line = line.strip().split()
            if list_line[0] == ":":
                if verbose:
                    print("\n" + line + "\n")
                if old_cat is not None:
                    all_cat.append(old_cat)
                    all_cat_scores.append(old_score)
                    if raw:
                        all_cat_totals.append(old_total)
                    else:
                        all_cat_totals.append(old_total * 10)
                    old_cat = list_line[1]
                    old_score = 0
                    old_total = 0
                else:
                    old_cat = list_line[1]
                    old_score = 0
                    old_total = 0
            if all([word in word2index for word in list_line]):
                current_score = 0
                valid_tests += 1
                old_total += 1
                analogues = analogy(list_line[0],
                                    list_line[1],
                                    list_line[2],
                                    word2index,
                                    embeddings)[::-1]
                if raw:
                    if list_line[3] == analogues[9]:
                        current_score = 1
                else:
                    if list_line[3] in analogues:
                        current_score = analogues.index(list_line[3]) + 1
                old_score += current_score
                if verbose:
                    sys.stdout.write('\rline:{}|cat:{}|score:{}'.format(total_lines,
                                                                        old_cat,
                                                                        old_score))
                    sys.stdout.flush()
    all_cat.append(old_cat)
    all_cat_scores.append(old_score)
    if raw:
        all_cat_totals.append(old_total)
    else:
        all_cat_totals.append(old_total * 10)
    results = [cat + ": {0:.1f}% ({1}/{2})".format((score/total)*100,
                                                   score, total)
               for (cat, score, total) in zip(all_cat,
                                              all_cat_scores,
                                              all_cat_totals) if total != 0]
    if all_cat_totals == []:
        final_score = 0
        print("Every line has at least a word outside the vocabulary")
    else:
        final_score = np.sum(all_cat_scores) / np.sum(all_cat_totals)
    return final_score, results
