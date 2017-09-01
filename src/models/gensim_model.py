from WrapperModel import WrapperModel
import gensim
import pickle
import os
import sys
import inspect

try:
    from utils import prepare_corpus_folder
except ImportError:

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    from utils import prepare_corpus_folder


class Gensim(WrapperModel):

    def __init__(self, language):

        self.language = language

        self.size = 0

        self.model_name = 'Gensim'

    def train(self, path_to_corpus, prepare_corpus_func, **kwargs):

        corpus = prepare_corpus_func(path_to_corpus)

        # vector size
        size = kwargs.get('size', 100)

        self.size = size

        window = kwargs.get('window', 5)
        min_count = kwargs.get('min_count', 5)
        workers = kwargs.get('workers', 4)

        self.model = gensim.models.Word2Vec(corpus,
                                            size=size,
                                            window=window,
                                            min_count=min_count,
                                            workers=workers)

    def get_pickle(self):

        '''

        Function that saves a pickle file with the following dict:



        -- embeddings : the matrix of word embeddings

        -- word2index : a dict of the form word : index.



        '''

        word2index = {word: index for index, word in enumerate(list(self.model.wv.vocab))}

        # the following code creates the pickle
        # folder if it doesn't exists already
        # name of file will be in format model||wordvectorsize||corpuslanguage.p

        name_piece = self.model_name + str(self.size) + self.language + ".p"
        file_name = "pickles/" + name_piece

        file = open(file_name, 'wb')

        new_dict = {'word2index': word2index, 'embeddings': self.get_embeddings()}

        pickle.dump(new_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    def get_embeddings(self):

        '''
        Function that return embeddings generate by internal model


        :rtype: np.array

        '''

        return self.model[self.model.wv.vocab]


if __name__ == "__main__":

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)

    os.chdir(parentdir)

    model = Gensim('english')

    path = os.path.join(parentdir, 'corpora/toy-corpus-1')

    func = prepare_corpus_folder
    model.train(path, func)
    model.get_pickle()
