import unittest
import os
import sys
import inspect
import shutil

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils import run_test, load_embeddings
from eval.metrics import analogy


class MetricsTest(unittest.TestCase):
    """
    Class that test the functions of the metrics module.
    """
    @classmethod
    def setUpClass(cls):
        cls.filename = os.path.join(parentdir, 'pickles', "toy.pickle")
        cls.embeddings, cls.word2index = load_embeddings(cls.filename)

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


if __name__ == "__main__":
    run_test(MetricsTest,
             "\n=== Running tests for metric functions ===\n")
