import numpy as np
import random
import tensorflow as tf
import time
import os
import string
from collections import Counter, deque
from random import randint


class DataReader(object):
    """
    Class to read and manipulate text.
    """
    def __init__(self,
                 path=None,
                 punctuation=False,
                 write_vocab=False):
        """
        :type path: string >>> path to text
        :type punction: boolean
        :type write_vocab: boolean
        """

        self.path = path
        self.punctuation = punctuation
        self.write_vocab = write_vocab

    def read_text(self):
        """
        Given a path to a txt file 'path' this function
        reads each line of the file and stores each word in a list words.
        'punctuation' is a parameter to control if we want the punctuation
        of the text to be captured by this reading or not.

        :type path: string
        :type punctuation: boolean
        :rtype: list of strings
        """

        dic_trans = {key: None for key in string.punctuation}
        translator = str.maketrans(dic_trans)
        words = []
        with open(self.path) as inputfile:
            for line in inputfile:
                line = line.lower()
                if not self.punctuation:
                    line = line.translate(translator)

                words.extend(line.strip().split())
        return words

    def build_vocab(self, words, vocab_size):
        """
        Given one list of words 'words' and
        one int 'vocab_size' this functions constructs
        one list of (word, frequency) named 'count' of size vocab_size
        (only the vocab_size - 1 most frequent words are here, the rest will
        be discarded as 'UNK'). This function returns also two dicts
        'word2index' and 'index_to_word' to translate the words in
        indexes and vice-versa.
        The parameter 'write_vocab' controls if you want to creat a file
        'vocab_1000.tsv' for vector vizualization in Tensorboard.

        :type words: list of strings
        :type vocab_size: int
        :type write: boolean
        :rtype count: list of tuples -> (str,int)
        :rtype word2index: dictionary
        :rtype index2word: dictionary
        """
        count = [("UNK", 0)]
        most_frequent_words = Counter(words).most_common(vocab_size - 1)
        count.extend(most_frequent_words)
        word2index = {}
        index = 0

        if self.write_vocab:
            path = os.path.dirname(__file__)
            path = os.path.join(path, 'vocab_1000.tsv')
            f = open(path, "w")

        for word, _ in count:
            word2index[word] = index

            if index < 1000 and self.write_vocab:
                f.write(word + "\n")

            index += 1

        if self.write_vocab:
            f.close()

        index2word = dict(zip(word2index.values(), word2index.keys()))
        return count, word2index, index2word

    def process_data(self, vocab_size=50000):
        """
        This function transform the text "words" into a list
        of numbers according to the dictionary word2index.
        It also modifies the frequency counter 'count' to
        count the frequency of the word 'UNK'.

        :type words: list of strings
        :type count: list of tuples -> (str,int)
        :type word2index: dictionary
        :type index_to_word: list of dictionary
        :rtype data: list of ints
        :rtype count: list of tuples -> (str,int)

        :rtype data: list of ints
        :rtype count: list of tuples -> (str,int)
        :rtype word2index: dictionary
        :rtype index2word: list of dictionary
        """
        words = self.read_text()
        self.count, self.word2index, self.index2word = self.build_vocab(words,
                                                                        vocab_size)
        self.data = []
        unk_count = 0
        for word in words:
            index = self.word2index.get(word, 0)

            if not index:
                unk_count += 1

            self.data.append(index)

        self.count[0] = ('UNK', unk_count)

    def batch_generator(self,
                        batch_size,
                        num_skips,
                        skip_window,
                        data_index):

        """
        This functions goes thought the processed text 'data' (starting at
        the point 'data_index') and at each step creates a reading window
        of size 2 * skip_window + 1. The word in the center of this
        window will be the center word and it is stored in the array
        'batch'; this function also chooses at random one of the remaining
        words of the window and store it in the array 'labels'. The
        parameter num_skips controls how many times we will use the same center
        word. After all this processing the point in the text has changed, so
        this function also return the number 'data_index'.



        :type batch_size: int
        :type num_skips: int
        :type skip_window: int
        :type data_index: int
        :type data: list of ints
        :rtype data_index: int
        :rtype batch: np array -> [shape = (batch_size), dtype=np.int32]
        :rtype labels: np array -> [shape = (batch_size,1), dtype=np.int32]
        """
        if batch_size % num_skips != 0:
            raise ValueError(
                """batch_size ({0}) should be a multiple of num_skips ({1})""".format(batch_size, num_skips))
        if num_skips > 2 * skip_window:
            raise ValueError(
                """num_skips ({0}) should be less or equal than twice
                the value of skip_window ({1})""".format(num_skips, skip_window))

        data_size = len(self.data)
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1
        reading_window = deque(maxlen=span)
        for _ in range(span):
            reading_window.append(self.data[data_index])
            data_index = (data_index + 1) % data_size
        for i in range(int(batch_size / num_skips)):
            target = skip_window
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = randint(0, span - 1)
                targets_to_avoid.append(target)
                center_word = reading_window[skip_window]
                context_word = reading_window[target]
                batch[i * num_skips + j] = center_word
                labels[i * num_skips + j, 0] = context_word
            reading_window.append(self.data[data_index])
            data_index = (data_index + 1) % data_size

        return data_index, batch, labels


def newlogname():
    log_basedir = './graphs'
    run_label = time.strftime('%d-%m-%Y_%H-%M-%S')  # e.g. 12-11-2016_18-20-45
    return os.path.join(log_basedir, run_label)


class Config():
    """
    Holds model hyperparams and data information.
    The config class is used to store various hyperparameters.
    SkipGramModel objects are passed a Config() object at
    instantiation.
    """
    def __init__(self,
                 vocab_size=50000,
                 batch_size=140,
                 embed_size=128,
                 skip_window=1,
                 num_skips=2,
                 num_sampled=64,
                 lr=1.0,
                 std_param=0.01,
                 init_param=(1.0, 1.0),
                 num_steps=100001,
                 show_step=2000,
                 verbose_step=10000,
                 valid_size=16,
                 valid_window=100):
        """
        :type vocab_size: int >>> size of the vocabulary
        :type batch_size: int >>> size of the batch
        :type embed_size: int >>> size of the word embeddings
        :type skip_window: int >>> size of the context window
        :type num_skip: int >>> number of times a same center word is used
        :type num_sampled: int >>> number of samples for negativ sampling
        :type lr: float >>> learning rate
        :rtype std_param: float >>> parameter to define the standart deviation
                                    of the softmax weights
                                    std = 1.0/(embed_size**std_param)
        :rtype init_param: (float,float) >>> params to set up the uniform
                                             distribuition of the
                                             initialization of the
                                             word embeddings
        :type num_steps: int >>> number of training steps
        :type show_steps: int >>> steps to show the loss during training
        :type verbose_step: int >>> steps to show some valid examples
                                  during training
        :type valid_size: int >>> number of valid words to show in verbose_step
        :type valid_window: int >>> range of words to choose the valid words
        """
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.skip_window = skip_window
        self.num_skips = num_skips
        self.num_sampled = num_sampled
        self.lr = lr
        self.std_param = std_param
        self.init_param = init_param
        self.num_steps = num_steps
        self.show_step = show_step
        self.verbose_step = verbose_step
        self.valid_size = valid_size
        self.valid_window = valid_window
        self.valid_examples = np.array(random.sample(range(self.valid_window),
                                                     self.valid_size))


class SkipGramModel:
    """
    The Skipgram model. This class only instatiates
    the tensorflow graph for the model.
    """
    def __init__(self, config):
        """
        :type config: Config
        """
        self.logdir = newlogname()
        self.config = config
        self.vocab_size = self.config.vocab_size
        self.embed_size = self.config.embed_size
        self.batch_size = self.config.batch_size
        self.num_sampled = self.config.num_sampled
        self.lr = self.config.lr
        self.std_param = self.config.std_param
        self.init_param = self.config.init_param
        self.valid_examples = self.config.valid_examples
        self.build_graph()

    def create_placeholders(self):
        """
        Create placeholder for the models graph
        """
        with tf.name_scope("words"):
            self.center_words = tf.placeholder(tf.int32,
                                               shape=[self.batch_size],
                                               name='center_words')
            self.targets = tf.placeholder(tf.int32,
                                          shape=[self.batch_size, 1],
                                          name='target_words')
            self.valid_dataset = tf.constant(self.valid_examples,
                                             dtype=tf.int32)

    def create_weights(self):
        """
        Create all the weights and bias for the models graph
        """
        emshape = (self.vocab_size, self.embed_size)
        eminit = tf.random_uniform(emshape,
                                   -self.init_param[0],
                                   self.init_param[1])
        self.embeddings = tf.Variable(eminit, name="embeddings")

        with tf.name_scope("softmax"):
            Wshape = (self.vocab_size, self.embed_size)
            bshape = (self.vocab_size)
            std = 1.0 / (self.config.embed_size ** self.std_param)
            Winit = tf.truncated_normal(Wshape, stddev=std)
            binit = tf.zeros(bshape)
            self.weights = tf.get_variable("weights",
                                           dtype=tf.float32,
                                           initializer=Winit)
            self.biases = tf.get_variable("biases",
                                          dtype=tf.float32,
                                          initializer=binit)

    def create_loss(self):
        """
        Create the loss function of the model
        """
        with tf.name_scope("loss"):
            self.embed = tf.nn.embedding_lookup(self.embeddings,
                                                self.center_words,
                                                name='embed')
            self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(self.weights,
                                                                  self.biases,
                                                                  self.targets,
                                                                  self.embed,
                                                                  self.num_sampled,
                                                                  self.vocab_size))

    def create_optimizer(self):
        """
        Create the optimization of the model
        """
        with tf.name_scope("train"):
            opt = tf.train.AdagradOptimizer(self.lr)
            self.optimizer = opt.minimize(self.loss)

    def create_valid(self):
        """
        Create the valid vectors for comparison
        """
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings),
                                     1, keep_dims=True))
        self.normalized_embeddings = self.embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings,
                                                  self.valid_dataset)
        self.similarity = tf.matmul(valid_embeddings,
                                    tf.transpose(self.normalized_embeddings))

    def create_summaries(self):
        """
        Create the summary
        """
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        """
        Build the graph for our model
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.create_placeholders()
            self.create_weights()
            self.create_loss()
            self.create_optimizer()
            self.create_valid()
            self.create_summaries()


def run_training(model, data, verbose=False, visualization=False, debug=False):
    """
    Function to train the model. We use the parameter "verbose" to show
    some the words during training; "visualization" adds ternsorboard
    visualization; if "debug" is True then the return will be the duration of
    the training and the mean of the loss, and if "debug" is False this
    function returns the matrix of word embeddings.

    :type model: SkipGramModel
    :type data: Datareader
    :type verbose: boolean
    :type visualization: boolean
    :type debug: boolean
    :rtype duration: float
    :rtype avg_loss: float
    :rtype final_embeddings: np array -> [shape = (model.vocab_size,
                             model.embed_size), dtype=np.float32]
    """
    logdir = model.logdir
    batch_size = model.config.batch_size
    num_skips = model.config.num_skips
    skip_window = model.config.skip_window
    valid_examples = model.config.valid_examples
    num_steps = model.config.num_steps
    show_step = model.config.show_step
    verbose_step = model.config.verbose_step
    data_index = 0
    with tf.Session(graph=model.graph) as session:
        tf.global_variables_initializer().run()
        ts = time.time()
        if verbose:
            print("Initialized")
        if visualization:
            print("\n&&&&&&&&& For TensorBoard visualization type &&&&&&&&&&&")
            print("\ntensorboard  --logdir={}\n".format(logdir))
        average_loss = 0
        total_loss = 0
        if visualization:
            writer = tf.summary.FileWriter(logdir, session.graph)
        for step in range(num_steps):
            data_index, batch_data, batch_labels = data.batch_generator(batch_size,
                                                                        num_skips,
                                                                        skip_window,
                                                                        data_index)
            feed_dict = {model.center_words: batch_data,
                         model.targets: batch_labels}
            _, l, summary = session.run([model.optimizer,
                                         model.loss,
                                         model.summary_op],
                                        feed_dict=feed_dict)
            average_loss += l
            total_loss += l
            if visualization:
                writer.add_summary(summary, global_step=step)
                writer.flush()
            if step % show_step == 0:
                if step > 0:
                    average_loss = average_loss / show_step
                    if verbose:
                        print("Average loss at step", step, ":", average_loss)
                    average_loss = 0
            if step % verbose_step == 0 and verbose:
                sim = model.similarity.eval()
                for i in range(model.config.valid_size):
                    valid_word = data.index2word[valid_examples[i]]
                    top_k = 8
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log = "Nearest to %s:" % valid_word
                    for k in range(top_k):
                        close_word = data.index2word[nearest[k]]
                        log = "%s %s," % (log, close_word)
                    if verbose:
                        print(log)

        final_embeddings = model.normalized_embeddings.eval()

    te = time.time()
    duration = te - ts
    avg_loss = total_loss / num_steps
    if debug:
        return duration, avg_loss
    else:
        return final_embeddings


def process_text_data(file_path, vocab_size):
    """
    This function is responsible for preprocessing the text data we will use to
    train our model. It will perform the following steps:

    * Create an word array for the file we have received. For example, if our
      text is:

        'I want to learn wordvec to do cool stuff'

    It will produce the following array:

        ['I', 'want', 'to', 'learn', 'wordvec', 'to', 'do', 'cool', 'stuff']

    * Create the frequency count for every word in our array:

       [('I', 1), ('want', 1), ('to', 2), ('learn', 1), ('wordvec', 1),
        ('do', 1), ('cool', 1), ('stuff', 1)]

    * With the count array, we choose as our vocabulary the words with the
      highest count. The number of words will be decided by the variable
      vocab_size.

    * After that we will create a dictionary to map a word to an index and an
      index to a word:

      index2word: {0: 'I', 1: 'want', 2: 'to', 3: 'learn', 4: 'wordvec',
                   5: 'do', 6: 'cool', 7: 'stuff'}
      word2index: {'I': 0, 'want': 1, 'to': 2, 'learn': 3, 'wordvec': 4,
                   'do': 5, 'cool': 6, 'stuff': 7}

      Both of these dictionaries are based on the words provided by the count
      array.

    * Finally, we will transform the words array to a number array, using the
      word2vec dictionary.

      Therefore, our words array:

      ['I', 'want', 'to', 'learn', 'wordvec', 'to', 'do', 'cool', 'stuff']

      Will be translated to:

      [0, 1, 2, 3, 4, 2, 5, 6, 7]

      If a word is not present in the word2index array,
      it will be considered an
      unknown word. Every unknown word will be mapped to the same index.
    """
    my_data = DataReader(file_path)
    my_data.process_data(vocab_size)
    return my_data
