from .WrapperModel import WrapperModel
from .tensorflow_word2vec import word2vec

import pickle


class TFWord2Vec(WrapperModel):

    def __init__(self, language, model_name, window_size, embedding_size,
                 epochs_to_train):
        self.language = language
        self.model_name = model_name
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.epochs_to_train = epochs_to_train

    def train(self, path_to_corpus, prepare_corpus_func=None, **kwargs):
        self.model = word2vec.train_model(path_to_corpus,
                                          self.window_size,
                                          self.embedding_size,
                                          self.epochs_to_train)

    def get_pickle(self):
        word2index = self.model.word2index
        model_dict = {'word2index': word2index,
                      'embeddings': self.get_embeddings()}

        name_piece = self.model_name + str(self.size) + self.language + ".p"
        file_name = "pickles/" + name_piece

        with open(file_name, 'wb') as pkl_file:
            pickle.dump(model_dict, pkl_file)

    def get_embeddings(self):
        return self.model.embeddings
