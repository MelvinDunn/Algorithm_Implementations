"""

We describe an approach for unsupervised learning of a generic, 
distributed sentence encoder. Using the continuoity of text from 
books, we train an encoder-decoder model that tries to reconstruct 
the surrounding sentences of an encoded passage.
Sentences that share semantic and syntactic properties are thus mapped to similar vector representations. We next introduce a simple vocabulary expansion method to encode words that were not seen as part of training, allowing us to expand our vocabular to a million words. After training our model, we extract and evaluate our vectors with linear models on 8 tasks: semantic relatedness, paraphrase detection, image-sentence ranking, question-type 

Developing learning algorithms for distributed compositional semantics of words has long been a long-standing open problem at the intersection of language understanding and machine learning. In recent years, several approaches have ben developed for leanring composition operators that map word vectors to sentence vectors including recursive networks, recurrent networks, convolutional networks, and recursive-convolutional methods among others. All of these mehtods produce sentence representations that are passed to a supervised task and depend on a class label in order to backpropagate through the composition weights. Consequently, these methods learn high-quality sentence representations but are tuned only for their respective task. The paragraph vector of 7 is an alternative to the above models in that it can learn unsupervised sentence representations by introducing a distributed sentence indicator as part of a neural language model. The downside is at test time, infrerence needs to be performed to compute the new vector.

In this paper we abstract away from the composition methods themselves and consider an alternative loss function can be applied with any composition operator. We consider the following question: is there a task and a corresponding loss that will allow us to learn highly generic sentence vectors without a particularly supervised task in mind. Using word vector learning as inspiration, we propose an objective function that abstracts the skip-gram model of [8] to the sentence level. That is, instead of using a word to predict its surrounding context, we instead encode a sentence to predcit the sentences around it. Thus, any composition operator can be substituted as a sentence encoder and only the objective function becomes modifed. Figure 1 illustrates the model. We call our model skip-thoughts and vectors induced by our model are called skip-thought vectors.

Our model depends on having a training corpus of contiguous text. We chose to use a large collection of nevels neamely the Bookcorpus dataset for training our models. These are free vooks written by yet unpublished authors. The dataset contains 16 difference genroes, the table highlights the summary statistics of the book corpus. Along with narratives, books contain dialoughe, emotion and a wide range of interaction between characters. Furthermore, with a large enough collection the training set is not biased towards any particular domain or application.

of sentences from a model trained on the BookCorpus dataset. These results show that skip-thought vectors learn to accurately capture semantics and syntax of the sentences they encode.

We evaluate our vectors in a newly proposed setting: after leanring skip-thoughts, freeze the model and use the encoder as a generic feature extractor for arbitrary tasks. In our experiments we consider 8 tasks: semantic relatedness, paraphrase detection, image-sentence reanking and 5 standard classification benchmarks. In these experiments, we extract skip-thought vectors and train linear models to evaluate the representations directly, withought any additional fine-tuning. As it turns out, skip-thoughts yield generaic representations that perform robustly across all tasks considered.

One difficulty that arises with such an experimental setup is being able to construct a  large enough word vocabulay to encode arbitrary sentences. For example, a sentence form a Wikipedia article might contina nouns that are unlikely to appear in our book vocabulary. We solve this problem by learning a mapping that transfers word representations fromm one model to another. using pre-trained word2vec representations learned with a continuous bag-of-words model, we learn a linear mapping from a word in word2vec space to a word in the encoder's vocabulary space. The mapping is learned using all words that are shared between vocabularies. After training, any word that appears in word2vec can then get a vector in the encoder word embedding space.


2 Approach

2.1 Inducing skip-thought vectors

We treat skip-thoughts in the framework of the encoder-decoder models. That is, an encoder maps words toa  sentence vector and a decoder is used to generate the surrounding sentences. Encoder-Decoder models have gained a lot of traction for neural machine translation. In this setting, an encoder is used to map e.g. an English sentence into a vector. The decoder then conditions on this vector to generate a translation for the source English sentence. Several chouses of encoder-decoder pairs have been explored including ConvNet-RNN, RNN-RNN, LSTM-LSTM, the source sentence representation can also dynamilcally change through the use of an attenction mechnism to take into account only the relevant workds for translation at any given time. In our model, we use an RNN encoder with GRU actuvations and an RNN decoder with a conditional GRU. This model combination is nearly identical to the RNN encoder-decoder used in nueral machine translation. GRU has been shown to perform as well as LSTM on sequence modelling takss while being conceptually simpler. GRU units have only 2 gates and do not require the use of a cell. While we use RNNs for our model, any encoder and decoder can be used so long as we can backpropogate through it.

Assume we are given a sentence tuple(si-1, si, si+1). Let wti denote the t-th word for sentence si and let xti denote it's word embedding. We describe the model in three parts: the encoder, the decoder and objective function.


Encoder. Let w1....wiN be the words in a sentence si where N is the number of words in the sentence. At each time step, the encoder produces a hidden state hti which can be interpreted as the representation of the sequence w1...,wti. The hidden state hNi thus represents the full sentence.


Decoder. The decoder is a neural language model which coditions on the encoder
output hi. The computation is similar to that of the encoder except we
introduce matrices Cz, Cr, C that are used

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
		self.xt = xt

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
		xt = word_vectors[word_list[0][0]]
		W = np.random.randn(number_of_weights, 
			                xt.shape[0]) * 0.01
		U = np.random.randn(number_of_weights,
			                number_of_weights) * 0.01
		C = np.random.randn(number_of_weights,
			                number_of_weights) * 0.1
		self.xt = xt
		self.xt_minus_1 = np.zeros(xt.shape)
		# encoder weights
		self.Wr = W
		self.Wz = W
		self.W = W

		# decoder weights
		self.Wdr = W
		self.Wdz = W
		self.Wd = W

		# U matrix encoder
		self.Ur = U
		self.Uz = U
		self.U = U

		# U matrix decoder
		self.Udr = U
		self.Udz = U
		self.Ud = U

		# used to bias the update gate, 
		# reset gate and hidden state gate
		self.Cz = C
		self.Cr = C
		self.C = C

		# last proposed state.
		self.ht_minus_1 = np.zeros((number_of_weights,number_of_weights))



	def decoder(self, xt, hi):

		# reset gate
		rt = sigmoid(self.Wdr.dot(self.xt_minus_1) +
			         self.Udr.dot(self.ht_minus_1) +
			         self.Cr.dot(self.hi))

		# update gates
		zt = sigmoid(self.Wdz.dot(self.xt_minus_1)
			          + self.Udz.dot(self.ht_minus_1)
			          + self.Cz.dot(self.hi))

		# proposed state update
		hhat_t = tanh(self.Wd.dot(self.xt_minus_1)
			        + self.Ud.dot(rt * self.ht_minus_1)
			        + self.C.dot(hi))

		# denote the hidden state of the decoder
		ht_plus_1 = (1. - zt) * (self.ht_minus_1) + (zt * hhat_t)

		return ht_plus_1


	def forward(self):

		ht_minus_1 = 0
		# need to iterate through sentences.
		for i in range(len(self.word_list)):
			for k in range(len(self.word_list[i])):

				# current sentence
				st = (self.word_vectors[self.word_list[i][k]])

				"""
				# Don't index anything before 
				# the first word of the first sentence
				if i == 0:
					st_minus_1 = np.zeros(st.shape)	
				else:
					print(self.word_list[i][k])
					st_minus_1 = (self.word_vectors[self.word_list[i-1][k]])
				
				try:
					st_plus_1 = (self.word_vectors[self.word_list[i+1][k]])
				"""

				ht = self.encoder(st)
				

				if i is not 0:

					ht_minus_1 = self.decoder(st_minus_1, ht)

				st_minus_1 = st

				print(ht_minus_1)



if __name__ == "__main__":
	# word_vectors = get_word_vectors('/data/converted_word2vec.txt')
	sentence_list = [["no","sir","do","not","have","."],
	                 ["give", "admiral", "diamond", "."],
	                 ["was", "thanos", "really", "mad", "."],
	                 ["do","not","wear","the","band","shirt","at","that","bands","show","."]]
	# skip = SkipThought(sentence_list, word_vectors, 5)
	print(Decoder(np.random.randn(50,1),np.random.randn(10,10),10).decoder(np.random.randn(50,1),np.random.randn(50,1),
		                                            np.random.randn(10,10)))


