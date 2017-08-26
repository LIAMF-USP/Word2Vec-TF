import unittest
import os
import sys
import inspect
import shutil
import linecache

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils import run_test, load_embeddings
from eval.metrics import analogy, naive_analogy_score
from eval.Evaluator import Evaluator
from eval.ModelComparison import ModelComparison


class EvalTest(unittest.TestCase):
    """
    Class that test the classes and functions from the metrics module.
    """
    @classmethod
    def setUpClass(cls):
        cls.toy_pickle1 = os.path.join(parentdir, 'pickles', "toy1.pickle")
        cls.toy_pickle2 = os.path.join(parentdir, 'pickles', "toy2.pickle")
        cls.embeddings, cls.word2index = load_embeddings(cls.toy_pickle1)
        cls.toy_analogy_path = os.path.join(parentdir,
                                            'analogies',
                                            "toy-ptbr.txt")
        cls.pt_analogy_path = os.path.join(parentdir,
                                           'analogies',
                                           "questions-words-ptbr.txt")
        cls.list_of_names = ["toy1", "toy2"]
        cls.list_of_pickles = [cls.toy_pickle1, cls.toy_pickle2]

    # @classmethod
    # def tearDown(cls):
    #     experiments_path = os.path.join(currentdir, "experiments")
    #     if os.path.exists(experiments_path):
    #         shutil.rmtree(experiments_path)

    def test_analogy(self):
        """
        Function to test if the analogy function can perform
        a basic analogy
        """
        similar_words = analogy('irmão',
                                'irmã',
                                'ele',
                                EvalTest.word2index,
                                EvalTest.embeddings)
        self.assertEqual(similar_words[0], "ela",
                         msg="\nsimilar_words= \n {}".format(similar_words))

    def test_analogy_score(self):
        """
        Function to test the naive_analogy_score function
        """
        score, result_list, _ = naive_analogy_score(EvalTest.word2index,
                                                    EvalTest.embeddings,
                                                    EvalTest.toy_analogy_path,
                                                    verbose=False)
        score *= 100
        self.assertEqual(score, 100,
                         msg="\nresults = \n {}".format(result_list))

    def test_Evaluator(self):
        """
        Function to test the metrics of the Evaluator
        """
        toy1_eval = Evaluator(EvalTest.toy_pickle1, EvalTest.pt_analogy_path)
        toy2_eval = Evaluator(EvalTest.toy_pickle2, EvalTest.pt_analogy_path)
        precision1, raw_score1, score1 = toy1_eval.get_metrics()
        precision2, raw_score2, score2 = toy2_eval.get_metrics()
        rounded_precision1 = round(precision1, 6)
        rounded_raw_score1 = round(raw_score1, 6)
        rounded_score1 = round(score1, 6)
        rounded_precision2 = round(precision2, 6)
        rounded_raw_score2 = round(raw_score2, 6)
        rounded_score2 = round(score2, 6)
        self.assertAlmostEqual(rounded_precision1,
                               0.000114,
                               places=5,
                               msg="precision = {}".format(rounded_precision1))
        self.assertAlmostEqual(rounded_raw_score1,
                               1.0,
                               places=3,
                               msg="precision = {}".format(rounded_raw_score1))
        self.assertAlmostEqual(rounded_score1,
                               1.0,
                               places=3,
                               msg="precision = {}".format(rounded_raw_score1))
        self.assertAlmostEqual(rounded_precision2,
                               0.023048,
                               places=3,
                               msg="precision = {}".format(rounded_precision2))
        self.assertAlmostEqual(rounded_raw_score2,
                               0.071604,
                               places=3,
                               msg="precision = {}".format(rounded_raw_score2))
        self.assertAlmostEqual(rounded_score2,
                               0.297778,
                               places=3,
                               msg="precision = {}".format(rounded_score2))

        # cls.df, cls.results = compare_models(cls.list_of_names,
        #                                      cls.list_of_pickles,
        #                                      cls.analogy_path,
        #                                      verbose=False)
    def test_compare_models(self):
        """
        Comparing two "models". 'toy1.pickle' has a matrix of word embeddings
        with less words than 'toy2.pickle'. So when we apply the analogy
        evaluation on these both embeddings, 'toy2' will pass in more
        analogy tests than 'toy1'.
        """
        ModelComparison(EvalTest.list_of_names,
                        EvalTest.list_of_pickles,
                        EvalTest.pt_analogy_path)
        # best_one = EvalTest.df.nlargest(1, 'Score*Preci')
        # result = list(best_one["Model Name"])[0]
        # self.assertEqual(result,
        #                  "toy2",
        #                  msg="\ndf = \n {}".format(EvalTest.df.to_string()))

    # def test_save_comparison(self):
    #     """
    #     Comparing two "models". 'toy1.pickle' has a matrix of word embeddings
    #     with less words than 'toy2.pickle'. So when we apply the analogy
    #     evaluation on these both embeddings, 'toy2' will pass in more
    #     analogy tests than 'toy1'.
    #     """
    #     filename = save_comparison(EvalTest.df,
    #                                EvalTest.results,
    #                                verbose=False)
    #     self.assertEqual(linecache.getline(filename, 7),
    #                      '===The best model is:===\n')
    #     self.assertEqual(linecache.getline(filename, 8),
    #                      '\n')
    #     self.assertEqual(linecache.getline(filename, 9),
    #                      '  Model Name  Precision     Score  Score*Preci\n')
    #     self.assertEqual(linecache.getline(filename, 10),
    #                      '1       toy2   0.023048  0.297778     0.006863\n')


if __name__ == "__main__":
    run_test(EvalTest,
             "\n=== Running tests for the eval module ===\n")
