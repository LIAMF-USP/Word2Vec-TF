import pandas as pd
import sys
import os

try:
    from utils import load_embeddings, timeit
except ImportError:

    import inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    from utils import load_embeddings, timeit


class ModelComparison:
    """
    Class that evaluates one model.
    It receives the model as a pickle file and creates a score based on the
    analogy test in eval_path.

    :type pickle_path: str
    :type eval_path: str
    :type encoding: str
    """
    def __init__(self,
                 list_of_model_names,
                 list_of_pickle_paths,
                 eval_path,
                 encoding="utf8"):

        self.eval_path = eval_path
        self.encoding = encoding
        self.embeddings, self.word2index = load_embeddings(pickle_path)
        self.top_results = min(self.embeddings.shape[0] - 2, 10)

    def compare_models(list_of_model_names,
                       list_of_pickle_paths,
                       eval_path,
                       verbose=True,
                       raw=False):
        """
        Given a list of model names, a list of pickles and an evaluation file,
        this function stores all the information given by the function
        analogy_score in a DataFrame. Here we calculate another metric
        'Score*Preci' which is the product bethween the score and the precision
        of a model. The ideia is that a good model has both higher precision
        (contemplate more words) and  higher score (accuracy in the analogy test)

        :type list_of_model_names: list
        :type list_of_pickle_paths: list
        :type eval_path: str
        :type verbose: boolean
        :type raw: boolean
        :rtype df: pd DataFrame
        :rtype results: list of dict

        """
        size_condition = len(list_of_model_names) == len(list_of_pickle_paths)
        assert size_condition, "model names and pickle paths: diferente sizes"
        results = []
        all_observations = []
        for name, path in zip(list_of_model_names, list_of_pickle_paths):
            embeddings, word2index = load_embeddings(path)
            score, result, precision,raw_score = analogy_score(word2index,
                                                     embeddings,
                                                     eval_path,
                                                     verbose=verbose,
                                                     raw=raw)
        observation = {}
        observation['Model Name'] = name
        observation['Raw_Score'] = raw_score
        observation['Score'] = score
        observation['Precision'] = precision
        observation['Score*Preci'] = score * precision
        all_observations.append(observation)
        results.append(result)
    dataframe = pd.DataFrame(all_observations)
    results = {name: result for name, result in zip(list_of_model_names,
                                                    results)}
    return dataframe, results


def save_comparison(dataframe, results, verbose=True):
    """
    Save the model comparison in a txt file.

    :type dataframe: pd DataFrame
    :type results: list of dict
    :type verbose: boolean
    :rtype: str
    """
    experiments_path = os.path.join(os.getcwd(), "experiments")
    if not os.path.exists(experiments_path):
        os.mkdir("experiments")
    experiment_name = "experiment_" + get_date_and_time() + ".txt"
    filename = os.path.join(experiments_path, experiment_name)
    with open(filename, "w") as file:
        file.write("===The results are:===\n\n")
        file.write(dataframe.to_string())
        best_one = dataframe.nlargest(1, 'Score*Preci')
        file.write("\n\n===The best model is:===\n\n")
        file.write(best_one.to_string())
        file.write("\n")
        for key in results.keys():
            file.write("\n\n===Detailed results for {}===\n".format(key))
            for info in results[key]:
                file.write("\n" + info)
    if verbose:
        print("You can find the saved file in {}".format(filename))
    return filename