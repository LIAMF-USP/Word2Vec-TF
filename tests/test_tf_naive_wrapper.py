import unittest
import os
import shutil

from src.models.naive_model import NaiveTfWord2Vec


class NaiveTFWord2VecTest(unittest.TestCase):

    @classmethod
    def tearDown(cls):
        currentdir = os.getcwd()
        pickles_path = os.path.join(currentdir, "pickles")
        graphs_path = os.path.join(currentdir, "graphs")
        if os.path.exists(pickles_path):
            shutil.rmtree(pickles_path)
        if os.path.exists(graphs_path):
            shutil.rmtree(graphs_path)

    def setUp(self):
        language = 'pt-br'
        model_name = 'naive'
        window_size = 1
        embedding_size = 10
        vocab_size = 243
        num_steps = 101
        show_step = 50

        self.model_wrapper = NaiveTfWord2Vec(language,
                                             model_name,
                                             window_size,
                                             embedding_size,
                                             vocab_size,
                                             num_steps=num_steps,
                                             show_step=show_step)

    def test_train(self):
        path_to_corpus = os.path.join('tests', 'test_corpora', 'test.txt')
        self.model_wrapper.train(path_to_corpus)

        expected_embeddings_size = (243, 10)
        embeddings = self.model_wrapper.get_embeddings()
        self.assertEqual(expected_embeddings_size, embeddings.shape)

        path = self.model_wrapper.get_pickle()
        self.assertTrue(os.path.exists(path))
