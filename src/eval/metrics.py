import pickle
import heapq


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
