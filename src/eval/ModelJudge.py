import pandas as pd
import sys
import os
import inspect

try:
    from Evaluator import Evaluator
except ImportError:
    from eval.Evaluator import Evaluator
try:
    from utils import get_date_and_time
except ImportError:
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    from utils import get_date_and_time


class ModelJudge:
    """
    Class that compares different models
    using one list of model's name and one list of pickle.

    :type list_of_model_names: list of str
    :type list_of_pickle_path: list of str
    :type eval_path: str
    :type encoding: str
    """
    def __init__(self,
                 list_of_model_names,
                 list_of_pickle_paths,
                 eval_path,
                 encoding="utf8",
                 verbose=False):

        size_condition = len(list_of_model_names) == len(list_of_pickle_paths)
        assert size_condition, "model names and pickle paths: diferente sizes"
        self.list_of_model_names = list_of_model_names
        self.list_of_pickle_paths = list_of_pickle_paths
        self.eval_path = eval_path
        self.encoding = encoding
        self.date_and_time = get_date_and_time()
        self.verbose = verbose
        self.experiments_path = os.path.join(os.getcwd(), "experiments")

    def _build_dataframe(self):
        """
        Method to store all the information given by the class
        Evaluator in a DataFrame. Here we calculate another metric
        'Score*Preci' which is the product between score,
        raw_score and precision of a model. The ideia is
        that a good model has both higher precision (contemplate more words)
        and higher score (accuracy in the analogy test).
        """
        all_observations = []
        for name, path in zip(self.list_of_model_names,
                              self.list_of_pickle_paths):
            if self.verbose:
                print("\nEvaluating the model {}".format(name))
            evaluator = Evaluator(path, self.eval_path, self.encoding)
            precision, raw_score, score = evaluator.get_metrics()
            observation = {}
            observation['Name'] = name
            observation['Raw_Score'] = raw_score
            observation['Score'] = score
            observation['Precision'] = precision
            observation['Score*Preci'] = raw_score * score * precision
            all_observations.append(observation)
        self.dataframe = pd.DataFrame(all_observations)

    def _save_comparison_txt(self):
        """
        Save the model comparison in a txt file.
        """
        experiment_name = "experiment_" + self.date_and_time + ".txt"
        self.filename_txt = os.path.join(self.experiments_path,
                                         experiment_name)
        with open(self.filename_txt, "w") as file:
            file.write("===The results are:===\n\n")
            file.write(self.dataframe.to_string())
            self.best_df = self.dataframe.nlargest(1, 'Score*Preci')
            file.write("\n\n===The best model is:===\n\n")
            file.write(self.best_df.to_string())
            file.write("\n")
        if self.verbose:
            print("You can find the txt file in {}".format(self.filename_txt))

    def _save_comparison_csv(self):
        """
        Save the model comparison in a csv file.
        """
        experiment_name = "experiment_" + self.date_and_time + ".csv"
        self.filename_csv = os.path.join(self.experiments_path,
                                         experiment_name)
        self.dataframe.to_csv(self.filename_csv, index=False)
        if self.verbose:
            print("You can find the csv file in {}".format(self.filename_csv))

    def _plot_results(self):
        """
        Ploting the score results.
        """
        pass

    def compare(self):
        """
        Method for the client compare all models.
        """
        if not os.path.exists(self.experiments_path):
            os.mkdir("experiments")
        self._build_dataframe()
        self._save_comparison_txt()
        self._save_comparison_csv()
        self._plot_results()

    def get_best(self):
        """
        assas
        """
        return list(self.best_df["Name"])[0]
