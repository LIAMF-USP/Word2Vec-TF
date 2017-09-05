"""
EXPERIMENT 1 & 2

Experiment with the models gensim, and the official tensorflow implementation
with different embedding sizes.

We use both a corpus in portuguese as a corpus in english with preprocessing.
"""
import os
from train_functions import train_both_models_with_different_emb_sizes
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
emb_list = [80, 90, 100, 120, 200, 300, 500]
epochs_to_train = 5

# EXPERIMENT 1: portuguese

train_both_models_with_different_emb_sizes(pt_language,
                                           window_size,
                                           emb_list,
                                           epochs_to_train,
                                           pt_path_to_corpus,
                                           "experiment1")

# EXPERIMENT 2: english

train_both_models_with_different_emb_sizes(en_language,
                                           window_size,
                                           emb_list,
                                           epochs_to_train,
                                           en_path_to_corpus,
                                           "experiment2")
