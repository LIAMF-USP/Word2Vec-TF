from .WrapperModel import WrapperModel
from .Naive_W2V import word2vec
import pickle
import os


class NaiveTfWord2Vec(WrapperModel):

    def __init__(self,
                 language,
                 model_name,
                 window_size,
                 embedding_size,
                 vocab_size=50000,
                 batch_size=140,
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

        self.language = language
        self.model_name = model_name
        self.embedding_size = embedding_size
        self.config = word2vec.Config(vocab_size=vocab_size,
                                      batch_size=batch_size,
                                      embed_size=embedding_size,
                                      skip_window=window_size,
                                      num_skips=num_skips,
                                      num_sampled=num_sampled,
                                      lr=lr,
                                      std_param=std_param,
                                      init_param=init_param,
                                      num_steps=num_steps,
                                      show_step=show_step,
                                      verbose_step=verbose_step,
                                      valid_size=valid_size,
                                      valid_window=valid_window)
        self.model = word2vec.SkipGramModel(self.config)

    def train(self, path_to_corpus, prepare_corpus_func=None, **kwargs):
        self.data = word2vec.process_text_data(path_to_corpus,
                                               self.config.vocab_size)
        self.embeddings = word2vec.run_training(self.model, self.data)

    def get_pickle(self):
        word2index = self.data.word2index
        model_dict = {'word2index': word2index,
                      'embeddings': self.get_embeddings()}

        pickle_folder = os.path.join(os.getcwd(), "pickles")
        if not os.path.exists(pickle_folder):
            os.mkdir("pickles")
        prefix = self.model_name + str(self.embedding_size)
        name_piece = prefix + self.language + ".p"
        file_name = os.path.join(pickle_folder, name_piece)

        with open(file_name, 'wb') as pkl_file:
            pickle.dump(model_dict, pkl_file)
        return file_name

    def get_embeddings(self):
        try:
            return self.embeddings
        except AttributeError:
            print("You must train the model first!")
