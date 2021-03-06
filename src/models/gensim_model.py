from .WrapperModel import WrapperModel
import gensim
import pickle
import os
import sys
import inspect

try:
    from utils import prepare_corpus_file
except ImportError:

    almost_current = os.path.abspath(inspect.getfile(inspect.currentframe()))
    currentdir = os.path.dirname(almost_current)
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    from utils import prepare_corpus_file


class Gensim(WrapperModel):

    def __init__(self,
                 language,
                 model_name,
                 window_size,
                 embedding_size,
                 min_count=5,
                 workers=4):

        self.language = language
        self.model_name = model_name
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.min_count = min_count
        self.workers = workers

    def train(self, path_to_corpus):
        corpus = prepare_corpus_file(path_to_corpus)
        self.model = gensim.models.Word2Vec(corpus,
                                            size=self.embedding_size,
                                            window=self.window_size,
                                            min_count=self.min_count,
                                            workers=self.workers)

    def get_pickle(self):

        word2index = {word: index
                      for index, word in enumerate(list(self.model.wv.vocab))}
        new_dict = {'word2index': word2index,
                    'embeddings': self.get_embeddings()}

        pickle_folder = os.path.join(os.getcwd(), "pickles")
        if not os.path.exists(pickle_folder):
            os.mkdir("pickles")
        self.short_name = self.model_name + str(self.embedding_size)
        name_piece = self.short_name + self.language + ".p"
        file_name = os.path.join(pickle_folder, name_piece)

        with open(file_name, 'wb') as pkl_file:
            pickle.dump(new_dict, pkl_file)
        return file_name

    def get_embeddings(self):
        return self.model[self.model.wv.vocab]
