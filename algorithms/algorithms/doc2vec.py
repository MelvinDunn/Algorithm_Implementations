"""
CBOW implementation is used here.
"""

import numpy as np
import re

def get_data(filepath):
    f = open(filepath, 'r')
    message = f.read()
    return message

def clean_text(corpus):
	#splitting by the sentence here
    return [[re.sub('[^A-Za-z0-9]+', '', j) for j in i.split()] \
    for i in corpus.lower().split("\n")]

def create_vocabulary(corpus):
    #set will just output whatever order so it has to be sorted.
    return sorted(list(set(sum(corpus, []))))

def word_matrix(corpus):
	"""
	generate a word matrix from the query vector
	"""
	items = ([query_vector(corpus,words[0].split()) for words in corpus])
	return np.asarray(items)

def query_vector(document,words):
	"""
	The query vector is basically a one hot encoder for words in a document
	"""
	vocab = create_vocabulary(document)
	indices = [vocab.index(word) for word in words]
	empty = np.zeros(len(vocab))
	empty[[indices]] = 1.
	return empty

def get_paragraph_ids(corpus):
	return [key for key,value in enumerate(corpus)]

def softmax(x):
	#in the paper they also implement heirarchical softmax.
	expon = np.exp(x)
	return  expon / np.sum(expon)

def pv_dbow(corpus, window=1, iteration=1):
	#this matrix needs to be updated for every iteration.
	empty_h_matrix =  np.zeros(word_matrix(corpus).shape)
	for i in range(iteration):
		for document_id, document in enumerate(corpus):
			D_matrix = np.zeros(len(corpus))
			D_matrix[document_id] = 1.
			sample_word = (create_pv_dbow_output(document, window=window))
			one_hot_output = (query_vector(corpus, [sample_word]))
	return one_hot_output


def shallow_network(input_matrix, output_matrix, hidden_layers=10):
	#what are the dimensions for the input weights and biases
	#what are the dimensions for the output weights and biases


def create_pv_dbow_output(doc, window=2):
    outputs = []
    window_range = list(range(-window, window+1))
    #sample a random range
    item = 0
    window_item = np.random.randint(len(doc))
    for j in window_range:
    	if window_item == 0 and np.sign(j) == -1:
            pass
        else:
	        try:
	            item = doc[window_item+j]
	            outputs.append(item)
	        except IndexError:
	            pass
	try:
		sample_index = np.random.randint(len(outputs))
	except:
		sample_index = 0
    return outputs[sample_index]

if __name__ == "__main__":
    cleaned_corpus = (clean_text(get_data('data/steely_dan_deacon_blues_lyrics.txt')))
    print(cleaned_corpus)
    print(create_vocabulary(cleaned_corpus))
    print((word_matrix(cleaned_corpus)).shape)
    print(len(create_vocabulary(cleaned_corpus)))
    print(pv_dbow(cleaned_corpus))