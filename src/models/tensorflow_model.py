from .WrapperModel import WrapperModel
from .tensorflow_word2vec import word2vec
import pickle
import os


class TFWord2Vec(WrapperModel):

    def __init__(self, language, model_name, window_size, embedding_size,
                 epochs_to_train, encoding="utf8"):
        self.language = language
        self.model_name = model_name
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.epochs_to_train = epochs_to_train
        self.encoding = encoding

    def train(self, path_to_corpus, prepare_corpus_func=None):
        self.model = word2vec.train_model(path_to_corpus,
                                          self.window_size,
                                          self.embedding_size,
                                          self.epochs_to_train)

    def get_pickle(self):
        # word2index = self.model.word2index
        word2index = {k.decode(self.encoding): v
                      for k, v in self.model.word2index.items()}

        model_dict = {'word2index': word2index,
                      'embeddings': self.get_embeddings()}

        pickle_folder = os.path.join(os.getcwd(), "pickles")
        if not os.path.exists(pickle_folder):
            os.mkdir("pickles")
        self.short_name = self.model_name + str(self.embedding_size)
        name_piece = self.short_name + self.language + ".p"
        file_name = os.path.join(pickle_folder, name_piece)

        with open(file_name, 'wb') as pkl_file:
            pickle.dump(model_dict, pkl_file)
        return file_name

    def get_embeddings(self):
        return self.model.embeddings
