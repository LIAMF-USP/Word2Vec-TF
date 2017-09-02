import unittest
import os
import shutil
from src.models.gensim_model import Gensim
from src.utils import prepare_corpus_folder


class GensimWord2VecTest(unittest.TestCase):

    @classmethod
    def tearDown(cls):
        currentdir = os.getcwd()
        pickles_path = os.path.join(currentdir, "pickles")
        if os.path.exists(pickles_path):
            shutil.rmtree(pickles_path)

    def setUp(self):
        language = 'pt-br'
        embedding_size = 10
        window_size = 1
        self.model_wrapper = Gensim(language, embedding_size, window_size)

    def test_train(self):
        path_to_corpus = os.path.join('tests', 'test_corpora', 'test.txt')
        func = prepare_corpus_folder
        self.model_wrapper.train(path_to_corpus, func)

        expected_embeddings_size = (227, 10)
        embeddings = self.model_wrapper.get_embeddings()
        self.assertEqual(expected_embeddings_size, embeddings.shape)

        path = self.model_wrapper.get_pickle()
        self.assertTrue(os.path.exists(path))
