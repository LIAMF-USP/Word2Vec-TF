import unittest
import os
import shutil

from src.models.tensorflow_model import TFWord2Vec


class TFWord2VecTest(unittest.TestCase):

    @classmethod
    def tearDown(cls):
        currentdir = os.getcwd()
        pickles_path = os.path.join(currentdir, "pickles")
        if os.path.exists(pickles_path):
            shutil.rmtree(pickles_path)

    def setUp(self):
        language = 'pt-br'
        model_name = 'tf'
        window_size = 1
        embedding_size = 10
        epochs_to_train = 1

        self.model_wrapper = TFWord2Vec(language,
                                        model_name,
                                        window_size,
                                        embedding_size,
                                        epochs_to_train)

    def test_train(self):
        path_to_corpus = os.path.join('tests', 'test_corpora', 'test.txt')
        self.model_wrapper.train(path_to_corpus)

        expected_vocabulary_size = 243
        word2index = self.model_wrapper.model.word2index
        self.assertEqual(expected_vocabulary_size, len(word2index))

        expected_embeddings_size = (243, 10)
        embeddings = self.model_wrapper.get_embeddings()
        self.assertEqual(expected_embeddings_size, embeddings.shape)

        path = self.model_wrapper.get_pickle()
        self.assertTrue(os.path.exists(path))
