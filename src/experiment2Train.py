"""
EXPERIMENT 2

Experiment with the models gensim, tf and naive-tf with different
embedding sizes. The corpus is a portuguese text **with** preprossening.
"""
import os
from models.naive_model import NaiveTfWord2Vec
from models.tensorflow_model import TFWord2Vec
from models.gensim_model import Gensim
from utils import clean_text
import pickle

if not os.path.exists():
    path_to_raw_corpus = os.path.join('corpora', 'pt96.txt')
    clean_text(path_to_raw_corpus)

path_to_corpus = os.path.join('corpora', 'pt96CLEAN.txt')
pt_analogy_path = os.path.join('analogies', "questions-words-ptbr.txt")

pickles = []
names = []
language = '_ptC'
window_size = 5
EMB_LIST = [80, 90, 100, 120, 200, 300, 500]
size = len(EMB_LIST)
vocab_size = 50000
epochs_to_train = 15
naive_model_name = 'n'
tf_model_name = 'tf'
g_model_name = 'g'
verbose = True

for i, embedding_size in enumerate(EMB_LIST):
    print("{0}/{1}: Training word embeddings of size {2}".format(i,
                                                                 size,
                                                                 embedding_size))

    naive_model = NaiveTfWord2Vec(language,
                                  naive_model_name,
                                  window_size,
                                  embedding_size,
                                  vocab_size,
                                  verbose=verbose)

    print("\n====Training the naive tf model====\n")
    naive_model.train(path_to_corpus)
    pickles.append(naive_model.get_pickle())
    names.append(naive_model.name_piece)

    tf_model = TFWord2Vec(language,
                          tf_model_name,
                          window_size,
                          embedding_size,
                          epochs_to_train)

    print("\n====Training the official tf model====\n")
    tf_model.train(path_to_corpus)
    pickles.append(tf_model.get_pickle())
    names.append(tf_model.name_piece)

    g_model = Gensim(language,
                     g_model_name,
                     window_size,
                     embedding_size)

    print("\n====Training the Gensim model====\n")
    g_model.train(path_to_corpus)
    pickles.append(g_model.get_pickle())
    names.append(g_model.name_piece)

new_dict = {'names': names,
            'pickles': pickles}

pickle_folder = os.path.join(os.getcwd(), "pickles")

if not os.path.exists(pickle_folder):
    os.mkdir("pickles")

file_name = os.path.join("pickles", "experiment2.p")
with open(file_name, 'wb') as pkl_file:
    pickle.dump(new_dict, pkl_file)