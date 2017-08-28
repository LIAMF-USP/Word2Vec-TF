import pandas as pd
import sys
import os
import inspect
import matplotlib.pyplot as plt
import seaborn as sns

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
        self.verbose = verbose
        self.experiments_path = os.path.join(os.getcwd(), "experiments")
        self.date_and_time = get_date_and_time()
        self.folder_name = os.path.join(self.experiments_path,
                                        self.date_and_time)

    def _create_filename(self, filename):
        """
        Function that return a filename using the folder_name as a prefix.

        :type filename: str
        :rtype: str
        """
        return os.path.join(self.folder_name, filename)

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
        self.filename_txt = self._create_filename("results.txt")
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
        self.filename_csv = self._create_filename("data.csv")
        self.dataframe.to_csv(self.filename_csv, index=False)
        if self.verbose:
            print("You can find the csv file in {}".format(self.filename_csv))

    def _plot_metric(self, metric, title, filename):
        """
        Function to plot one metric and save it as the
        file "filename".

        :type metric: str
        :type title: str
        :type filename: str
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax = sns.pointplot(x="Name",
                           y=metric,
                           data=self.dataframe)
        fig.suptitle(title, fontsize=24, fontweight='bold')
        plt.savefig(filename)

    def compare(self):
        """
        Method for the client to compare all models.
        The results are saved in different forms:
        a txt file, a csv file and 4 images (png)
        """
        if not os.path.exists(self.experiments_path):
            os.mkdir("experiments")
        folder = os.path.join("experiments", self.date_and_time)
        if not os.path.exists(folder):
            os.mkdir(folder)
        self._build_dataframe()
        self._save_comparison_txt()
        self._save_comparison_csv()
        self.filename_precision = self._create_filename("precision.png")
        self.filename_raw_score = self._create_filename("raw_score.png")
        self.filename_score = self._create_filename("score.png")
        self.filename_score_preci = self._create_filename("score_preci.png")
        self._plot_metric("Precision",
                          "Precision per model",
                          self.filename_precision)
        self._plot_metric("Raw_Score",
                          "Raw Score per model",
                          self.filename_raw_score)
        self._plot_metric("Score",
                          "Score per model",
                          self.filename_score)
        self._plot_metric("Score*Preci",
                          "Raw Score * Score * Precision per model",
                          self.filename_score_preci)

    def get_best(self):
        """
        Function that return the name of the best model.
        Best here is define be the maximum value of the
        feature "Score*Preci"

        :rtype: str
        """
        try:
            return list(self.best_df["Name"])[0]
        except AttributeError:
            self.compare()
            return list(self.best_df["Name"])[0]
