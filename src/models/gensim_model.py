from Wrapper_Model import Wrapper_Model




class Gensim(Wrapper_Model):



	def __init__(self,language,**kwargs):



		self.language = language

		#vector size
		self.size = kwargs.get('size',100)


		self.window = kwargs.get('window',5)
		self.min_count = kwargs.get('min_count',5)
		self.workers = kwargs.get('workers',4)




	def train(self,path_to_corpus,prepare_corpus_func):


		corpus = prepare_corpus_func(path_to_corpus)


		import gensim

			

		self.model = gensim.models.Word2Vec(corpus,size = self.size,
												   window = self.window, 
												   min_count = self.min_count,
												   workers=self.workers)



	def get_pickle(self):

		'''

		Function that saves a pickle file with the following dict:



		-- embeddings : the matrix of word embeddings

		-- word2index : a dict of the form word : index.



		'''

		import pickle


		

		word2index = {word:index for index,word in enumerate(list(self.model.wv.vocab))}

		#the following code creates the pickle folder if it doesn't exists already

		import os

		file_name = "pickles/" + self.model_name + ".p"

		os.makedirs(os.path.dirname(file_name), exist_ok=True)




		file = open(file_name, 'wb')

		pickle.dump( {'word2index': word2index ,'embeddings':self.get_embeddings()} , file,protocol=pickle.HIGHEST_PROTOCOL )




	def get_embeddings(self):

		'''
		Function that return embeddings generate by internal model


		:rtype: np.array

		'''


		return self.model[self.model.wv.vocab]











if __name__ == "__main__":

	import unittest
	import os
	import sys
	import inspect

	currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
	parentdir = os.path.dirname(currentdir)
	sys.path.insert(0, parentdir)
	
	model = Gensim('english')



	path = os.path.join(parentdir, 'corpora/toy-corpus-1')
			
	func = model.prepare_corpus_folder

	model.train(path,func)




