"""
query expansion from 

http://anthology.aclweb.org/P/P16/P16-1035.pdf

Query Expansion with Locally-Trained Word Embeddings

This implementation will assume all documents in
the corpus come from the same domain / local embedding.

"""
import gensim
import numpy as np
import re
import os

PATH = os.path.dirname(os.path.abspath(__file__))

def stopwords():
    """
    A list of nltk's stopwords. I don't enjoy calling the library, because that's extra runtime.
    """
    
    return ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 
            'ourselves', 'you', "you're", "you've", "you'll", 
            "you'd", 'your', 'yours', 'yourself', 'yourselves', 
            'he', 'him', 'his', 'himself', 'she', "she's", 
            'her', 'hers', 'herself', 'it', "it's", 'its', 
            'itself', 'they', 'them', 'their', 'theirs', 
            'themselves', 'what', 'which', 'who', 'whom', 
            'this', 'that', "that'll", 'these', 'those', 'am', 
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 
            'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
            'because', 'as', 'until', 'while', 'of', 'at', 'by', 
            'for', 'with', 'about', 'against', 'between', 'into', 
            'through', 'during', 'before', 'after', 'above', 'below', 
            'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 
            'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', 
            "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 
            'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', 
            "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', 
            "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 
            'mightn', "mightn't", 'mustn', "mustn't", 'needn', 
            "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
            'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]



def expansion_term_weights_pq(global_or_local_embeddings, query_embedding):
	"""
	would be best if you just took a subset of
	global embeddings, aka a wikipedia or text8 corpus
	because multiplying large matrices like that might
	be computationally inefficient.
	"""
	return np.matmul(global_or_local_embeddings, global_or_local_embeddings.T) * query_embedding

def kullback_leibler_divergence(pq, pd):
	"""
	computes kullback leibler.
	equation 3 of the paper.
	added the softmax
	"""
	return softmax(pq * np.log(pq / pd))


def softmax(x):
	"""
	equation 5 in the paper. 
	refers to softmax function.
	"""
	return np.exp(x) / np.exp(x).sum()

def normalize(x):
	return x / np.linalg.norm(x)

def interpolated_language_model(pq_w, tiny_lambda=0.02):
	"""
	Fortunately, we can use information retrieval
	techniques to generate a query-specific
	set of topical documents. Specifically, we
	adopt a language modeling approach to do so
	(Croft and Lafferty, 2003). In this retrieval
	model, each document is represented as a maximum
	likelihood language model estimated
	from document term frequencies
	"""
	pq_plus_w = normalize(pq_w)
	return (tiny_lambda * pq_w) + ((1 - tiny_lambda)* pq_plus_w)

def query_expansion_main(fname):
	#word_vectors = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
	#word_vectors = gensim.models.Word2Vec.load(fname)
	word_vectors = word2vec_main(fname)
	return word_vectors


def word2vec_main(fname):
	input_file = PATH + fname
	file = open(input_file, 'r')
	documents = ((file.read().split('\n')))

	def remove_stopwords(words):
		"""
		Only allows alphanumeric characters, cleans the stopwords and splits them, respectively.
		"""
		words = re.sub(r'[^A-Za-z]', ' ', words)
		words = words.lower().split()
		return " ".join([word for word in words if word not in stopwords()])

	documents = [remove_stopwords(i).lower().split() for i in documents]

	print(documents)
	print(len(documents))

	model = gensim.models.Word2Vec(
	        documents,
	        size=300,
	        window=2,
	        min_count=1,
	        workers=4,
	        iter=10)

	model.train(documents, total_examples=len(documents), epochs=3)

	return model

if __name__ == "__main__":
	word_vectors = (query_expansion_main('/data/ohsumed.txt'))
	query_vector = (word_vectors.wv.get_vector("hiv"))
	whole_vector = (word_vectors.wv.syn0)
	print(word_vectors.wv.most_similar("hiv"))
	expansion_of_term_weights_UUtq = np.nan_to_num(np.matmul(whole_vector.T, whole_vector).dot(query_vector.reshape(query_vector.shape[0],1)))
	most_similar_index = (np.argmin(kullback_leibler_divergence(expansion_of_term_weights_UUtq, whole_vector.T)))
	print(word_vectors.wv.index2word[most_similar_index])
	most_similar = whole_vector[most_similar_index]
	pq_plus_w = (interpolated_language_model(most_similar))
	most_similar_index = (np.argmin(kullback_leibler_divergence(pq_plus_w, whole_vector)))
	print(word_vectors.wv.index2word[most_similar_index])
