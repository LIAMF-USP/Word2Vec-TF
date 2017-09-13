import pickle
import os
import sys
import inspect

almost_current = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(almost_current)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from models.tensorflow_model import TFWord2Vec
from models.gensim_model import Gensim


def train_both_models_with_different_emb_sizes(language,
                                               window_size,
                                               emb_list,
                                               epochs_to_train,
                                               path_to_corpus,
                                               experiment_name):
    """
    Function to train the tensorflow model and the gensim model
    using diffent embedding sizes. After the training is done
    a pickle file is created containing the names of the models and
    the path for the respectives pickle files.

    :type language: str
    :type window_size: int
    :type emb_list: list of ints
    :type epochs_to_train: int
    :type path_to_corpus: str
    :type experiment_name: str
    """
    size = len(emb_list)
    pickles = []
    names = []
    for i, emb in enumerate(emb_list):
        print("{0}/{1}: Training word embeddings of size {2}".format(i + 1,
                                                                     size,
                                                                     emb))
        print("\n====Training the official tf model====\n")
        tf_model = TFWord2Vec(language,
                              'tf',
                              window_size,
                              emb,
                              epochs_to_train)

        tf_model.train(path_to_corpus)
        pickles.append(tf_model.get_pickle())
        names.append(tf_model.short_name)
        print("\n====Training the Gensim model====\n")
        g_model = Gensim(language,
                         'g',
                         window_size,
                         emb)

        g_model.train(path_to_corpus)
        pickles.append(g_model.get_pickle())
        names.append(g_model.short_name)

    new_dict = {'names': names,
                'pickles': pickles}

    pickle_folder = os.path.join(os.getcwd(), "pickles")

    if not os.path.exists(pickle_folder):
        os.mkdir("pickles")

    file_name = os.path.join("pickles", experiment_name + ".p")
    with open(file_name, 'wb') as pkl_file:
        pickle.dump(new_dict, pkl_file)


def train_both_models_with_different_window_sizes(language,
                                                  emb_size,
                                                  window_list,
                                                  epochs_to_train,
                                                  path_to_corpus,
                                                  experiment_name):
    """
    Function to train the tensorflow model and the gensim model
    using diffent window sizes. After the training is done
    a pickle file is created containing the names of the models and
    the path for the respectives pickle files.

    :type language: str
    :type emb_size: int
    :type window_list: list of ints
    :type epochs_to_train: int
    :type path_to_corpus: str
    :type experiment_name: str
    """
    size = len(window_list)
    pickles = []
    names = []
    for i, window_size in enumerate(window_list):
        print("{0}/{1}: window size = {2}".format(i + 1, size, window_size))
        sufix = "W" + str(window_size)
        print("\n====Training the official tf model====\n")
        tf_model = TFWord2Vec(language,
                              'tf' + sufix,
                              window_size,
                              emb_size,
                              epochs_to_train)

        tf_model.train(path_to_corpus)
        pickles.append(tf_model.get_pickle())
        names.append(tf_model.short_name)
        print("\n====Training the Gensim model====\n")
        g_model = Gensim(language,
                         'g' + sufix,
                         window_size,
                         emb_size)

        g_model.train(path_to_corpus)
        pickles.append(g_model.get_pickle())
        names.append(g_model.short_name)

    new_dict = {'names': names,
                'pickles': pickles}

    pickle_folder = os.path.join(os.getcwd(), "pickles")

    if not os.path.exists(pickle_folder):
        os.mkdir("pickles")

    file_name = os.path.join("pickles", experiment_name + ".p")
    with open(file_name, 'wb') as pkl_file:
        pickle.dump(new_dict, pkl_file)
