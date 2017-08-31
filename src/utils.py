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

        with open(dir_path + '/' + filename, "r") as text_file:

            corpus = corpus + simple_clean(text_file.read())

    return corpus
