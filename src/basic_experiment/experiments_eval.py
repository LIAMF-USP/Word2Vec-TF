"""
Evaluate all the embeddings produce by the experiments

Before run this program you shuld run

bash experiments_script.sh
"""
import os
import pickle
import sys
import inspect
import matplotlib
matplotlib.use('Agg')


almost_current = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(almost_current)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from eval.ModelJudge import ModelJudge

pt_analogy_path = os.path.join(parentdir,
                               'analogies',
                               "questions-words-ptbr.txt")

en_analogy_path = os.path.join(parentdir,
                               'analogies',
                               "questions-words.txt")


def judge_experiments(file_name, analogy_path, experiment_name):
    """
    Given a pickle file called "file_name" with a list of models
    and a list of pickles, this fuctions takes one analogy text
    to evaluate all models.

    :type file_name: str
    :type analogy_path: str
    :type experiment_name: str
    """

    with open(file_name, "rb") as pkl_file:
        d = pickle.load(pkl_file)
        pass

    names = d['names']
    pickles = d['pickles']

    judge = ModelJudge(names,
                       pickles,
                       analogy_path,
                       verbose=True,
                       experiment_name=experiment_name)
    judge.compare()


file_name1 = os.path.join("pickles", "experiment1.p")
file_name2 = os.path.join("pickles", "experiment2.p")
file_name3 = os.path.join("pickles", "experiment3.p")
file_name4 = os.path.join("pickles", "experiment4.p")
file_name5 = os.path.join("pickles", "experiment5.p")
file_name6 = os.path.join("pickles", "experiment6.p")
file_name7 = os.path.join("pickles", "experiment7.p")
file_name8 = os.path.join("pickles", "experiment8.p")


judge_experiments(file_name1, pt_analogy_path, "experiment1")
judge_experiments(file_name2, en_analogy_path, "experiment2")
judge_experiments(file_name3, pt_analogy_path, "experiment3")
judge_experiments(file_name4, en_analogy_path, "experiment4")
judge_experiments(file_name5, pt_analogy_path, "experiment5")
judge_experiments(file_name6, en_analogy_path, "experiment6")
judge_experiments(file_name7, pt_analogy_path, "experiment7")
judge_experiments(file_name8, en_analogy_path, "experiment8")
