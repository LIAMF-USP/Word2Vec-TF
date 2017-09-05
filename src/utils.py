import re
import pickle
import time
import unittest

timing = {}


def get_time(f, args=[]):
    """
    After using timeit we can get the duration of the function f
    when it was applied in parameters args. Normally it is expected
    that args is a list of parameters, but it can be also a single parameter.

    :type f: function
    :type args: list
    :rtype: float
    """
    if type(args) != list:
        args = [args]
    key = f.__name__
    if args != []:
        key += "-" + "-".join([str(arg) for arg in args])
    return timing[key]


def timeit(index_args=[]):

    def dec(method):
        """
        Decorator for time information
        """

        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            timed.__name__ = method.__name__
            te = time.time()
            fkey = method.__name__
            for i, arg in enumerate(args):
                if i in index_args:
                    fkey += "-" + str(arg)
            timing[fkey] = te - ts
            return result
        return timed
    return dec


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
    with open(pickle_path, "rb") as pkl_file:
        d = pickle.load(pkl_file)
        pass

    embeddings = d['embeddings']
    word2index = d['word2index']
    del d

    return embeddings, word2index


def simple_clean(text):
    '''
    Function that performs simple cleanup in text.
    Remove posting header, split by sentences and words,
    keep only letters

    :type text: str
    :rtype str
    '''

    lines = re.split('[?!.:]\s', re.sub('^.*Lines: \d+', '',
                                        re.sub('\n', ' ', text)))
    return [re.sub('[^a-zA-Z]', ' ', line).lower().split() for line in lines]


def prepare_corpus_file(text_path):
    '''
    Helper function that takes one text files
    in a folder and creates a list of lists with all words in each file.

    :type text_path: str
    '''

    corpus = []
    with open(text_path, "r") as text_file:
        corpus = simple_clean(text_file.read())

    return corpus


def prepare_corpus_folder(dir_path):
    '''
    Helper function that takes all text files
    in a folder and creates a list of lists with all words in each file.

    :type text_path: str
    '''
    import os

    corpus = []

    for filename in os.listdir(dir_path):

        with open(dir_path + '/' + filename, "r") as text_file:

            corpus = corpus + simple_clean(text_file.read())

    return corpus

def clean_text(path):
    """
    Function that remove every link, every emoji with "EMOJI" and multiple
    spaces. It also puts every  word in the lower case format.

    :type path: str
    """
    new_path = path[:-4] + "CLEAN.txt"
    url = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    spaces = re.compile(' +')
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    with open(new_path, "w") as f:
        for line in open(path):
            line = line.lower()
            new_line = url.sub("LINK", line)
            new_line = emoji_pattern.sub("EMOJI", new_line)
            new_line = spaces.sub(" ", new_line)
            f.write(new_line)
