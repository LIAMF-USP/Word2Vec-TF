import unittest
import pandas as pd
import os
import shutil
import linecache
import matplotlib
matplotlib.use('Agg')
from src.utils import load_embeddings  # noqa
from src.eval.Evaluator import Evaluator  # noqa
from src.eval.ModelJudge import ModelJudge  # noqa
from src.models.gensim_model import Gensim  # noqa
from src.models.tensorflow_model import TFWord2Vec  # noqa


class EvalTest(unittest.TestCase):
    """
    Class that test the classes and functions from the eval module.
    """
    @classmethod
    def setUpClass(cls):
        cls.toy_pickle1 = os.path.join('tests', 'test_pickles', "toy1.pickle")
        cls.toy_pickle2 = os.path.join('tests', 'test_pickles', "toy2.pickle")
        cls.embeddings, cls.word2index = load_embeddings(cls.toy_pickle1)
        cls.pt_analogy_path = os.path.join('src', 'analogies',
                                           "questions-words-ptbr.txt")
        cls.list_of_names = ["toy1", "toy2"]
        cls.list_of_pickles = [cls.toy_pickle1, cls.toy_pickle2]
        cls.judge = ModelJudge(cls.list_of_names,
                               cls.list_of_pickles,
                               cls.pt_analogy_path)
        cls.best_model = cls.judge.get_best()

    @classmethod
    def tearDown(cls):
        currentdir = os.getcwd()
        experiments_path = os.path.join(currentdir, "experiments")
        pickles_path = os.path.join(currentdir, "pickles")
        if os.path.exists(experiments_path):
            shutil.rmtree(experiments_path)
        if os.path.exists(pickles_path):
            shutil.rmtree(pickles_path)

    def test_Evaluator(self):
        """
        Function to test the metrics of the Evaluator.
        with an older implementation of
        the eval module I got the following results:

           Name  Precision  Raw_Score     Score  Score*Preci
        0  toy1   0.000114   1.000000  1.000000     0.000114
        1  toy2   0.023066   0.071605  0.297778     0.000492

        So I am using these numbers to test the Evaluator class.
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

    def test_ModelJudge_pick_best(self):
        """
        Comparing two "models". 'toy1.pickle' has a matrix of word embeddings
        with less words than 'toy2.pickle'. When we apply the analogy
        evaluation on these both embeddings, 'toy2' will pass in more
        analogy tests than 'toy1'. So 'toy2' is the best model.
        """
        best_string = EvalTest.judge.best_df.to_string()
        self.assertEqual(EvalTest.best_model,
                         "toy2",
                         msg="\ndf = \n {}".format(best_string))

    def test_ModelJudge_train2log(self):
        """
        Testing if the ModelJudge is creating all the log files
        after the three basic models perform their training routine.

        **This test should always pass. It is a toy version of an experiment**

        """
        path_to_corpus = os.path.join('tests', 'test_corpora', 'test.txt')
        pickles = []
        names = []
        language = '_pt'
        window_size = 1
        embedding_size = 10
        epochs_to_train = 1
        tf_model_name = 'tf'
        g_model_name = 'g'

        tf_model = TFWord2Vec(language,
                              tf_model_name,
                              window_size,
                              embedding_size,
                              epochs_to_train)

        tf_model.train(path_to_corpus)
        pickles.append(tf_model.get_pickle())
        names.append(tf_model.short_name)

        g_model = Gensim(language,
                         g_model_name,
                         window_size,
                         embedding_size)

        g_model.train(path_to_corpus)
        pickles.append(g_model.get_pickle())
        names.append(g_model.short_name)

        judge = ModelJudge(names,
                           pickles,
                           EvalTest.pt_analogy_path)
        judge.compare()
        self.assertTrue(os.path.exists(judge.filename_txt))
        self.assertTrue(os.path.exists(judge.filename_csv))
        self.assertTrue(os.path.exists(judge.filename_precision))
        self.assertTrue(os.path.exists(judge.filename_raw_score))
        self.assertTrue(os.path.exists(judge.filename_score))
        self.assertTrue(os.path.exists(judge.filename_score_preci))

    def test_ModelJudge_txt(self):
        """
        Testing if the txt file has the right format
        """
        judge = ModelJudge(EvalTest.list_of_names,
                           EvalTest.list_of_pickles,
                           EvalTest.pt_analogy_path)
        judge.compare()
        filename = judge.filename_txt
        header = '===The best model is:===\n'
        df_header = '   Name  Precision  Raw_Score     Score  Score*Preci\n'
        df_result = '1  toy2   0.023066   0.071605  0.297778     0.000492\n'
        self.assertEqual(linecache.getline(filename, 7),
                         header)
        self.assertEqual(linecache.getline(filename, 8),
                         '\n')
        self.assertEqual(linecache.getline(filename, 9),
                         df_header)
        self.assertEqual(linecache.getline(filename, 10),
                         df_result)

    def test_ModelJudge_csv(self):
        """
        Testing if the csv file has the right format
        """
        judge = ModelJudge(EvalTest.list_of_names,
                           EvalTest.list_of_pickles,
                           EvalTest.pt_analogy_path)
        judge.compare()
        df = pd.read_csv(judge.filename_csv)
        best_df = df.nlargest(1, 'Score*Preci')
        best_model_from_csv = list(best_df["Name"])[0]
        self.assertEqual(EvalTest.best_model,
                         best_model_from_csv,
                         msg="\ndf = \n {}".format(best_df.to_string()))
