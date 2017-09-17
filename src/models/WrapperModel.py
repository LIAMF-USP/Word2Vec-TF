from abc import ABC, abstractmethod


class WrapperModel(ABC):
    """
    This Class implements a wrapper so that
    multiple implementations of word2vec
    can be used and easily compared
    """

    @abstractmethod
    def train(self, path_to_corpus, prepare_corpus_func):
        """
        Functions that trains the model with parameters passed on creation.

        :type path_to_corpus : str
        :type prepare_corpus_func : func

        Function to prepare corpus found on
        path_to_corpus for input in the model
        """
        pass

    @abstractmethod
    def get_pickle(self):
        """
        Function that saves a pickle file with the following dict:

        -- embeddings : the matrix of word embeddings
        -- word2index : a dict of the form word : index.

        and returns the path of the pickle

        :rtype: str
        """
        pass

    @abstractmethod
    def get_embeddings(self):
        """
        Function that return embeddings generate by internal model

        :rtype: np.array
        """
        pass
