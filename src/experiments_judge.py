import matplotlib
matplotlib.use('Agg')

import os
import pickle
from eval.ModelJudge import ModelJudge


def judge_experiments(file_name, analogy_path):

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

pt_analogy_path = os.path.join('analogies', "questions-words-ptbr.txt")

judge_experiments(file_name1, pt_analogy_path)
judge_experiments(file_name2, pt_analogy_path)
judge_experiments(file_name3, pt_analogy_path)
judge_experiments(file_name4, pt_analogy_path)
