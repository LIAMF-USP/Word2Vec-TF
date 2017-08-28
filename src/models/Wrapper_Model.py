

from abc import ABC, abstractmethod


class Wrapper_Model(ABC):

    '''

    This Class implements a wrapper so that multiple implementations of word2vec
    can be used and easily compared


    '''



    @staticmethod
    def simple_clean(text):
        '''
        Function that performs simple cleanup in text


        :type text: str
        :rtype str

        '''

        import re
        """Remove posting header, split by sentences and words, keep only letters"""
        lines = re.split('[?!.:]\s', re.sub('^.*Lines: \d+', '', re.sub('\n', ' ', text)))
        return [re.sub('[^a-zA-Z]', ' ', line).lower().split() for line in lines]


    @staticmethod
    def prepare_corpus_folder(dir_path):
        '''

        Helper function that takes all text files in a folder and creates a list of lists with all words in each file.


        :type dir_path: str


        '''

        import os

        corpus = []

        for filename in os.listdir(dir_path):

            with open( dir_path + '/' + filename, "r") as text_file:

                corpus = corpus + Wrapper_Model.simple_clean(text_file.read())


        return corpus



    @abstractmethod
    def train(self,path_to_corpus,prepare_corpus_func,):
        '''

        Functions that trains the model with parameters passed on creation.


        :type path_to_corpus : str 
        


        :type prepare_corpus_func : func
        
        Function to prepare corpus found on path_to_corpus for input in the model

        '''

        pass



    @abstractmethod
    def get_pickle(self):

        '''

        Function that saves a pickle file with the following dict:



        -- embeddings : the matrix of word embeddings

        -- word2index : a dict of the form word : index.


        '''    
        pass

    @abstractmethod
    def get_embeddings(self):

        '''
        Function that return embeddings generate by internal model


        :rtype: np.array

        '''

        pass




if __name__ == "__main__":
    pass


