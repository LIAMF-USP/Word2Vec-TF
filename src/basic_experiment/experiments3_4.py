"""
EXPERIMENT 3 & 4

Experiment with the models gensim, and the official tensorflow implementation
with different window sizes.

We use both a corpus in portuguese as a corpus in english with preprocessing.
"""
import os
from train_functions import train_both_models_with_different_window_sizes
import sys
import inspect
import subprocess

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
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

pt_language = '_ptC'
en_language = '_enC'
window_size = 5
emb_size = 500
window_list = [2, 5, 10, 15, 20, 25]
epochs_to_train = 5

# EXPERIMENT 3: portuguese

train_both_models_with_different_window_sizes(pt_language,
                                              emb_size,
                                              window_list,
                                              epochs_to_train,
                                              pt_path_to_corpus,
                                              "experiment3")

# EXPERIMENT 4: english

train_both_models_with_different_window_sizes(en_language,
                                              emb_size,
                                              window_list,
                                              epochs_to_train,
                                              pt_path_to_corpus,
                                              "experiment4")
