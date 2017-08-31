import numpy as np
import tensorflow as tf
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


class Evaluator:
    """
    Class that evaluates one model.
    It receives the model as a pickle file and creates a score based on the
    analogy test in eval_path.

    :type pickle_path: str
    :type eval_path: str
    :type encoding: str
    """
    def __init__(self,
                 pickle_path,
                 eval_path,
                 encoding="utf8"):

        self.eval_path = eval_path
        self.encoding = encoding
        self.embeddings, self.word2index = load_embeddings(pickle_path)
        self.top_results = min(self.embeddings.shape[0] - 2, 10)

    def _read_analogies(self):
        """
        Reads through the analogy question file.
        If every word in the analogy line is in the vocabulary
        we store this line in index form.

        builds:
            analogy_questions: a [n, 4] numpy array containing
                               the analogy question's word ids.
                               where n is the number of
                               questions not skipped from the
                               analogy question file.
        """
        questions = []
        questions_skipped = 0
        with open(self.eval_path, "rb") as analogy_f:
            for line in analogy_f:
                if line.startswith(b":"):
                    continue
                words = line.strip().lower().split(b" ")
                ids = [self.word2index.get(w.strip().decode(self.encoding))
                       for w in words]
                if None in ids or len(ids) != 4:
                    questions_skipped += 1
                else:
                    questions.append(np.array(ids))
        self.valid_questions = len(questions)
        total_questions = self.valid_questions + questions_skipped
        self.precision = self.valid_questions / total_questions
        self.analogy_questions = np.array(questions, dtype=np.int32)

    def _build_top_k(self):
        """
        Uses a tensorflow graph to build the top k closest words
        to the vector resulting from (c + b - a).
        Remember, each valid line of the analogy file is of the form

        a b c d

        And we are trying to check if d is the closest word
        to (c + b - a) using the cosine similarity.

        Since the indices of words a and b appear in the top_k matrix
        we need to take a matrix of size k+2.
        """
        graph = tf.Graph()
        with graph.as_default():
            normalized_emb = tf.nn.l2_normalize(self.embeddings, 1)
            a_emb = tf.gather(normalized_emb,
                              self.analogy_questions[:, 0])
            b_emb = tf.gather(normalized_emb,
                              self.analogy_questions[:, 1])
            c_emb = tf.gather(normalized_emb,
                              self.analogy_questions[:, 2])

            target = c_emb + (b_emb - a_emb)

            dist = tf.matmul(target,
                             normalized_emb,
                             transpose_b=True)
            _, pred_idx = tf.nn.top_k(dist, self.top_results + 2)

        with tf.Session(graph=graph) as sess:
            self.top_k = sess.run(pred_idx)

    def _build_clean_top_k(self):
        """
        Cleaning the top k matrix. Here we remove all the indices
        form the words b and c from the top k matrix
        """
        clean = []
        for i, line in enumerate(self.top_k):
            b_c = list(self.analogy_questions[i][1:3])
            new = []
            for idx in line:
                if idx not in b_c:
                    new.append(idx)
            if len(new) > self.top_results:
                new = line[:self.top_results]
            clean.append(np.array(new))
        self.clean_top_k = np.array(clean)

    @timeit([0])
    def get_metrics(self):
        """
        Returns 3 metrics:

            precision: questions not skipped / total number of questions

            raw_score: number of times we got a exact result
                       / number of valid questions

            score: sum(position of the word d in the inverse top k answer)
                       / number of valid questions * 10

        The main difference between score and raw_score is that
        the score metric tries to capture how close is the d word
        from (c + b - a) using numbers from [0,10] such that
        10 is the maximum point (d is the closest word from the
        vocabulary) and 0 is the lowest point (d is not even in the
        top 10 closest words).

        :rtype precision: float
        :rtype raw_score: float
        :rtype score: float

        """
        self._read_analogies()
        self._build_top_k()
        self._build_clean_top_k()
        all_d_idx = self.analogy_questions[:, 3]
        comparison = self.clean_top_k[:, 0] == all_d_idx
        exact_result = sum(comparison)

        raw_score = exact_result / self.valid_questions
        position_list = []
        for i, line in enumerate(np.flip(self.clean_top_k, 1)):
            try:
                position = np.where(line == all_d_idx[i])[0][0] + 1
            except IndexError:
                position = 0
            position_list.append(position)
        score = np.sum(position_list) / (self.valid_questions * 10)

        return self.precision, raw_score, score
