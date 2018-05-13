"""
First attempt at skip-thought.

Interesting idea.

https://arxiv.org/pdf/1506.06726.pdf
"""


import numpy as np
import gensim
import os

PATH = os.path.dirname(os.path.abspath(__file__))

def sigmoid(x):
	return 1. / (1. + np.exp(-x))


def tanh(z):
	e_z = np.exp(z)
	e_z_minus = np.exp(-z)
	return (e_z - e_z_minus) / (e_z - e_z_minus)

def get_word_vectors(fname):
	input_file = PATH + fname
	word_vectors = gensim.models.KeyedVectors.load_word2vec_format(input_file)
	return word_vectors


class Encoder:

	def __init__(self, xt, number_of_weights):

		W = np.random.randn(number_of_weights, 
			                xt.shape[0]) * 0.01
		U = np.random.randn(number_of_weights,
			                number_of_weights) * 0.01
		C = np.random.randn(number_of_weights,
			                number_of_weights) * 0.1

		# current word vector input

		# encoder weights
		self.Wr = W
		self.Wz = W
		self.W = W

		# U matrix encoder
		self.Ur = U
		self.Uz = U
		self.U = U		


	def encoder(self, xt, ht_minus_1):
		# reset gate
		rt = sigmoid(self.Wr.dot(xt)
			              + self.Ur.dot(ht_minus_1))
		# update gate
		zt = sigmoid(self.Wz.dot(xt)
			              + self.Uz.dot(ht_minus_1))

		# proposed state update at time t
		hhat_t = tanh(self.W.dot(xt) 
			          + self.U.dot(rt 
			          * ht_minus_1))
		# comonenent wise product is hadamard product \
		# is [1,1,1] * [1,2,3] == [1,2,3]
		ht = (1. - zt) * ht_minus_1 + (zt * hhat_t)		
		return ht


class Decoder:

	def __init__(self, xt, ht_minus_1, number_of_weights):

		W = np.random.randn(number_of_weights, 
			                xt.shape[0]) * 0.01
		U = np.random.randn(number_of_weights,
			                number_of_weights) * 0.01
		C = np.random.randn(number_of_weights,
			                number_of_weights) * 0.1

		# current word vector input
		self.xt = xt

		# the previous state, 
		# remember that a decoder comes before this
		self.ht_minus_1 = ht_minus_1

		# decoder weights
		self.Wdr = W
		self.Wdz = W
		self.Wd = W

		# U matrix decoder
		self.Udr = U
		self.Udz = U
		self.Ud = U	

		# used to bias the update gate, 
		# reset gate and hidden state gate
		self.Cz = C
		self.Cr = C
		self.C = C

	def decoder(self, xt, xt_minus_1, hi):

		# reset gate
		rt = sigmoid(self.Wdr.dot(xt_minus_1) +
			         self.Udr.dot(self.ht_minus_1) +
			         self.Cr.dot(hi))

		# update gates
		zt = sigmoid(self.Wdz.dot(xt_minus_1)
			          + self.Udz.dot(self.ht_minus_1)
			          + self.Cz.dot(hi))

		# proposed state update
		hhat_t = tanh(self.Wd.dot(xt_minus_1)
			        + self.Ud.dot(rt * self.ht_minus_1)
			        + self.C.dot(hi))

		# denote the hidden state of the decoder
		ht_plus_1 = (1. - zt) * (self.ht_minus_1) + (zt * hhat_t)

		return ht_plus_1


class SkipThought:
	
	def __init__(self, word_list, word_vectors, number_of_weights):
		

		self.word_list = word_list
		self.word_vectors = word_vectors
		self.number_of_weights = number_of_weights
		self.encoding = Encoder(np.random.randn(50,1) * 0.01,number_of_weights)
		self.decoder_before = Decoder(np.random.randn(50,1) * 0.01, np.random.randn(10,10) * 0.01, number_of_weights)
		# self.decoder_after = Decoder(np.random.randn(50,1) * 0.01, np.random.randn(50,1) * 0.01, np.random.randn(10,10))
		self.ht_minus_1 = ht_minus_1 = np.random.randn(self.number_of_weights, self.number_of_weights) * 0.01
		self.ht = np.random.randn(self.number_of_weights, self.number_of_weights) * 0.01
		self.ht_plus_1 = np.random.randn(self.number_of_weights, self.number_of_weights) * 0.01
	def forward(self):
		"""
		Pretty much just wrong, but hey, I tried.
		"""
		
		# need to iterate through sentences.

		# get the longest sentence length.
		max_len = max(len(l) for l in self.word_list)

		for i in range(len(self.word_list)):
			for k in range(max_len):
				if i > 0:
					try:
						self.ht_plus_1 = (self.decoder_before.decoder(self.word_vectors[self.word_list[i][k]],self.word_vectors[self.word_list[i-1][k]], self.ht))
					except IndexError:
						pass
				try:
					self.ht = (self.encoding.encoder(self.word_vectors[self.word_list[i][k]], self.ht_minus_1))
					print()
				except IndexError:
					pass

				if i is not max_len:
					try:
						self.ht_plus_1 = (self.decoder_before.decoder(self.word_vectors[self.word_list[i][k]],self.word_vectors[self.word_list[i-1][k]], self.ht))
					except IndexError:
						pass

		return self.ht





if __name__ == "__main__":
	word_vectors = get_word_vectors('/data/converted_word2vec.txt')
	sentence_list = [["no","sir","do","not","have","."],
	                 ["we", "all", "live", "in", "a", "yellow", "submarine", "."],
	                 ["was", "thanos", "really", "mad", "."],
	                 ["do","not","wear","the","band","shirt","at","that","bands","show","."]]
	print(SkipThought(sentence_list, word_vectors, 10).forward())
	#print(Decoder(np.random.randn(50,1),np.random.randn(10,10),10).decoder(np.random.randn(50,1),np.random.randn(50,1),
	#	                                            np.random.randn(10,10)))


