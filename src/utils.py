import pickle
import time
import unittest


def get_date_and_time():
    """
    Function to create an unique label
    using the date and time.

    :rtype: str
    """
    return time.strftime('%d-%m-%Y_%H-%M-%S')


def run_test(testClass, header):
    """
    Function to run all the tests from a class of tests.

    :type testClass: unittest.TesCase
    :type header: str
    """
    print(header)
    suite = unittest.TestLoader().loadTestsFromTestCase(testClass)
    unittest.TextTestRunner(verbosity=2).run(suite)


def get_revert_dict(some_dict):
    """
    Reverting a dict

    :type some_dict: dict
    :rtype: dict
    """
    reverse_dict = {v: k for k, v in some_dict.items()}
    return reverse_dict


def load_embeddings(pickle_path):
    """
    Function that receives a path to a pickle file. We assume that
    in this file we have two objects:

    -- embeddings : the matrix of word embeddings

    -- word2index : a dict of the form word : index.

    :type pickle_path: str
    :rtype: np.array, dict
    """
    with open(pickle_path, "rb") as file:
        d = pickle.load(file)
        pass

    embeddings = d['embeddings']
    word2index = d['word2index']
    del d

    return embeddings, word2index


class word2vec_wrapper:

    '''

    This Class implements a wrapper so that multiple implementations of word2vec
    can be used and easily compared


    '''




    def __init__(self,model_name,language):
        '''

        :type model_name: str

        :type language: str


        '''

        try:
            possible_names = ["gensim","word2vec_TF"]
            model_name in possible_names
        except:
            print("Model not implemented!")


        self.model_name = model_name

        self.language   = language



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


    def prepare_corpus_folder(dir_path):
        '''

        Helper function that takes all text files in a folder and creates a list of lists with all words in each file.


        :type dir_path: str


        '''

        import os

        corpus = []

        for filename in os.listdir(dir_path):

            with open( dir_path + '/' + filename, "r") as text_file:

                corpus = corpus + word2vec_wrapper.simple_clean(text_file.read())


        return corpus


    def train(self,path_to_corpus,prepare_corpus_func):


        corpus = prepare_corpus_func(path_to_corpus)


        if(self.model_name == 'gensim'):

            import gensim


            self.model = gensim.models.Word2Vec(corpus,size=100, window=5, min_count=5, workers=4)



    def get_pickle(self):

        '''

        Function that saves a pickle file with the following dict:



        -- embeddings : the matrix of word embeddings

        -- word2index : a dict of the form word : index.



        '''

        import pickle


        if(self.model_name == 'gensim'):

            word2index = {word:index for index,word in enumerate(list(self.model.wv.vocab))}

            #the following code creates the pickle folder if it doesn't exists already

            import os

            file_name = "pickles/" + self.model_name + ".p"

            os.makedirs(os.path.dirname(file_name), exist_ok=True)




            file = open(file_name, 'wb')

            pickle.dump( {'word2index': word2index ,'embeddings':self.get_embeddings()} , file,protocol=pickle.HIGHEST_PROTOCOL )




    def get_embeddings(self):

        '''
        Function that return embeddings generate by internal model


        :rtype: np.array

        '''


        if(self.model_name == 'gensim'):
            return self.model[self.model.wv.vocab]
