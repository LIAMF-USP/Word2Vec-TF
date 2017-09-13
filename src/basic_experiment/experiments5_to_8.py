"""
EXPERIMENT 5, 6, 7, & 8

Experiment with the models gensim, and the official tensorflow implementation
with different window sizes.

We use both a corpus in portuguese as a corpus in english
with and without preprocessing.
"""
import os
from train_functions import train_both_models_with_different_window_sizes
import sys
import inspect
import subprocess

almost_current = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(almost_current)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils import clean_text

en_path_to_raw_corpus = os.path.join('corpora', 'text8.txt')
pt_path_to_raw_corpus = os.path.join('corpora', 'pt96.txt')


en_path_to_corpus = os.path.join('corpora', 'text8CLEAN.txt')
pt_path_to_corpus = os.path.join('corpora', 'pt96CLEAN.txt')

en_condition = os.path.exists(en_path_to_raw_corpus)
pt_condition = os.path.exists(pt_path_to_raw_corpus)

if not (en_condition and pt_condition):
    pro = subprocess.Popen(["bash", "download_corpora.sh"])
    pro.wait()


if not os.path.exists(en_path_to_corpus):
    clean_text(en_path_to_raw_corpus)


if not os.path.exists(pt_path_to_corpus):
    clean_text(pt_path_to_raw_corpus)

pt_languageR = 'pR'
en_languageR = 'eR'
pt_languageC = 'pC'
en_languageC = 'eC'

window_size = 5
emb_size = 500
window_list = [2, 5, 10, 15, 20, 25]
epochs_to_train = 5

# EXPERIMENT 5: portuguese raw

train_both_models_with_different_window_sizes(pt_languageR,
                                              emb_size,
                                              window_list,
                                              epochs_to_train,
                                              pt_path_to_raw_corpus,
                                              "experiment5")

# EXPERIMENT 6: english raw

train_both_models_with_different_window_sizes(en_languageR,
                                              emb_size,
                                              window_list,
                                              epochs_to_train,
                                              en_path_to_raw_corpus,
                                              "experiment6")

# EXPERIMENT 7: portuguese clean

train_both_models_with_different_window_sizes(pt_languageC,
                                              emb_size,
                                              window_list,
                                              epochs_to_train,
                                              pt_path_to_corpus,
                                              "experiment7")

# EXPERIMENT 8: english clean

train_both_models_with_different_window_sizes(en_languageC,
                                              emb_size,
                                              window_list,
                                              epochs_to_train,
                                              en_path_to_corpus,
                                              "experiment8")
