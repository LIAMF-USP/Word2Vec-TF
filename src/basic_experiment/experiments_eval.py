"""
Evaluate all the embeddings produce by the experiments

Before run this program you shuld run

bash experiments_script.sh
"""
import matplotlib
matplotlib.use('Agg')

import os
import pickle
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from eval.ModelJudge import ModelJudge

pt_analogy_path = os.path.join(parentdir,
                               'analogies',
                               "questions-words-ptbr.txt")

en_analogy_path = os.path.join(parentdir,
                               'analogies',
                               "questions-words.txt")


def judge_experiments(file_name, analogy_path):
    """
    Given a pickle file called "file_name" with a list of models
    and a list of pickles, this fuctions takes one analogy text
    to evaluate all models.

    :type file_name: str
    :type analogy_path: str
    """

    with open(file_name, "rb") as pkl_file:
        d = pickle.load(pkl_file)
        pass

    names = d['names']
    pickles = d['pickles']

    judge = ModelJudge(names,
                       pickles,
                       analogy_path,
                       verbose=True)
    judge.compare()


file_name1 = os.path.join("pickles", "experiment1.p")
file_name2 = os.path.join("pickles", "experiment2.p")
file_name3 = os.path.join("pickles", "experiment3.p")
file_name4 = os.path.join("pickles", "experiment4.p")


judge_experiments(file_name1, pt_analogy_path)
judge_experiments(file_name2, en_analogy_path)
judge_experiments(file_name3, pt_analogy_path)
judge_experiments(file_name4, en_analogy_path)
