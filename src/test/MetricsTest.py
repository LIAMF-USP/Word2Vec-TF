import unittest
import os
import sys
import inspect
import shutil

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils import run_test, load_embeddings
from eval.metrics import analogy, analogy_score, compare_models


class MetricsTest(unittest.TestCase):
    """
    Class that test the functions of the metrics module.
    """
    @classmethod
    def setUpClass(cls):
        cls.toy_pickle1 = os.path.join(parentdir, 'pickles', "toy1.pickle")
        cls.toy_pickle2 = os.path.join(parentdir, 'pickles', "toy2.pickle")
        cls.embeddings, cls.word2index = load_embeddings(cls.toy_pickle1)
        cls.toy_analogy_path = os.path.join(parentdir,
                                            'analogies',
                                            "toy_ptbr.txt")
        cls.analogy_path = os.path.join(parentdir,
                                        'analogies',
                                        "questions-words-ptbr.txt")

    def test_analogy(self):
        """
        Function to test if all the sentences that are
        being generated are valid tweets
        """
        similar_words = analogy('irmão',
                                'irmã',
                                'ele',
                                MetricsTest.word2index,
                                MetricsTest.embeddings)
        self.assertEqual(similar_words[0], "ela",
                         msg="\nsimilar_words= \n {}".format(similar_words))

    def test_analogy_score(self):
        """
        Function to test if all the sentences that are
        being generated are valid tweets
        """
        score, result_list, _ = analogy_score(MetricsTest.word2index,
                                              MetricsTest.embeddings,
                                              MetricsTest.toy_analogy_path,
                                              verbose=False)
        score *= 100
        self.assertEqual(score, 100,
                         msg="\nresults = \n {}".format(result_list))

    def test_compare_models(self):
        """
        Comparing two "models". 'toy1.pickle' has a matrix of word embeddings
        with less words than 'toy2.pickle'. So when we apply the analogy
        evaluation on these both embeddings, 'toy2' will pass in more
        analogy tests than 'toy1'.
        """
        list_of_names = ["toy1", "toy2"]
        list_of_pickles = [MetricsTest.toy_pickle1, MetricsTest.toy_pickle2]
        df, _ = compare_models(list_of_names,
                               list_of_pickles,
                               MetricsTest.analogy_path,
                               verbose=False)
        best_one = df.nlargest(1, 'Score*Preci')
        result = list(best_one["Model Name"])[0]
        self.assertEqual(result,
                         "toy2",
                         msg="\ndf = \n {}".format(df.to_string()))

if __name__ == "__main__":
    run_test(MetricsTest,
             "\n=== Running tests for metric functions ===\n")
