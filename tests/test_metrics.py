import unittest
import os
import sys
import inspect
import shutil
import linecache


from src.utils import run_test, load_embeddings
from src.eval.metrics import analogy, analogy_score, compare_models, save_comparison


class MetricsTest(unittest.TestCase):
    """
    Class that test the functions of the metrics module.
    """
    @classmethod
    def setUpClass(cls):
        source_dir = 'src'
        cls.toy_pickle1 = os.path.join(source_dir, 'pickles', "toy1.pickle")
        cls.toy_pickle2 = os.path.join(source_dir, 'pickles', "toy2.pickle")
        cls.embeddings, cls.word2index = load_embeddings(cls.toy_pickle1)
        cls.toy_analogy_path = os.path.join(source_dir,
                                            'analogies',
                                            "toy_ptbr.txt")
        cls.analogy_path = os.path.join(source_dir,
                                        'analogies',
                                        "questions-words-ptbr.txt")
        cls.list_of_names = ["toy1", "toy2"]
        cls.list_of_pickles = [cls.toy_pickle1, cls.toy_pickle2]
        cls.df, cls.results = compare_models(cls.list_of_names,
                                             cls.list_of_pickles,
                                             cls.analogy_path,
                                             verbose=False)

    @classmethod
    def tearDown(cls):
        experiments_path = 'experiments'
        if os.path.exists(experiments_path):
            shutil.rmtree(experiments_path)

    def test_analogy(self):
        """
        Function to test if the analogy function can perform
        a basic analogy
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
        Function to test the analogy_score function
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
        best_one = MetricsTest.df.nlargest(1, 'Score*Preci')
        result = list(best_one["Model Name"])[0]
        self.assertEqual(result,
                         "toy2",
                         msg="\ndf = \n {}".format(MetricsTest.df.to_string()))

    def test_save_comparison(self):
        """
        Comparing two "models". 'toy1.pickle' has a matrix of word embeddings
        with less words than 'toy2.pickle'. So when we apply the analogy
        evaluation on these both embeddings, 'toy2' will pass in more
        analogy tests than 'toy1'.
        """
        filename = save_comparison(MetricsTest.df,
                                   MetricsTest.results,
                                   verbose=False)
        self.assertEqual(linecache.getline(filename, 7),
                         '===The best model is:===\n')
        self.assertEqual(linecache.getline(filename, 8),
                         '\n')
        self.assertEqual(linecache.getline(filename, 9),
                         '  Model Name  Precision     Score  Score*Preci\n')
        self.assertEqual(linecache.getline(filename, 10),
                         '1       toy2   0.023048  0.297778     0.006863\n')


if __name__ == "__main__":
    run_test(MetricsTest,
             "\n=== Running tests for metric functions ===\n")
