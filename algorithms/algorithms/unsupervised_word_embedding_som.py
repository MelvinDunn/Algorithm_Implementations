"""
Unsupervised Text Classification and Search using
Word Embeddings on a Self-Organizing Map

https://pdfs.semanticscholar.org/e06e/3ce4611010e3ac835189cab04f20106eb62d.pdf
"""

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
import numpy as np
import re
from sklearn.cluster import KMeans
from gensim.scripts.glove2word2vec import glove2word2vec
import logging
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


def remove_stopwords(words):
    """
    Only allows alphanumeric characters, cleans the stopwords and splits them, respectively.
    """
    words = re.sub(r'[^A-Za-z]', ' ', words)
    words = words.lower().split()
    return [word for word in words if word not in stopwords()]


def remove_stopwords_from_corpus(corpus):
    return [[" ".join(remove_stopwords(words[0]))] for words in corpus]


def combine_corpus(corpus):
    return sum(shape_corpus_for_vocabulary(corpus), [])

def shape_corpus_for_vocabulary(corpus):
    return [words[0].split() for words in corpus]

def create_vocabulary(corpus):
    """
    creates a vocabulary for the word embeddings.
    sorts the vocabulary list in alphabetical order.
    the set function will return items in any random order.
    
    This function assumes the corpus is a list.
    """
    return sorted(list(set((combine_corpus(corpus)))))


def create_word_embedding(cleaned_corpus, vocabulary):
    """
    has the dimensionality of documents in the corpus x number of words in vocabulary
    be aware this is not one hot encoded, it's representing count of words
    """
    word_embedding_matrix = np.zeros((len(cleaned_corpus), (len(vocabulary))))
    cleaned_corpus_split = shape_corpus_for_vocabulary(cleaned_corpus)
    indices = ([(([vocabulary.index(j) for j in i])) for i in cleaned_corpus_split])
    for i in range(len(indices)):
        for j in indices[i]:
            word_embedding_matrix[i,j] +=  1.
    return word_embedding_matrix


def get_pretrained_word_vectors(corpus):
    glove_file = datapath(PATH + '/data/glove.6B.50d.txt')
    tmp_file = get_tmpfile("test_word2vec.txt")
    glove2word2vec(glove_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)
    word_vectors = model.wv
    del model
    #get a shaping error, and since knn is going to take the mean anyways, might as well take the mean.
    vectors_for_knn = ([np.mean(np.asarray([word_vectors[i] for i in corpus[j][0].split()]),axis=0) for j in range(len(corpus))])
    regular_vectors = ([np.sum(np.asarray([word_vectors[i] for i in corpus[j][0].split()]), axis=0) for j in range(len(corpus))])
    return vectors_for_knn, regular_vectors


def kmeans_pretrained_word_vectors(vectors):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(vectors)
    return kmeans.cluster_centers_


def large_object_for_counts_vectors_clusters(corpus):
    large_object = {}
    vocab = (create_vocabulary(corpus))
    large_object["count_vector"] = (create_word_embedding(corpus, vocab))
    del vocab
    vectors_for_knn, regular_vectors = get_pretrained_word_vectors(corpus)
    large_object["word_vectors"] =  np.asarray(regular_vectors)
    del regular_vectors
    large_object["cluster_centroids"] = kmeans_pretrained_word_vectors(vectors_for_knn)
    return large_object


def Kohonen_SOM(word_vectors_X, shape_N, neurons_K=10, iteration=10000, learning_rate_a=0.01, num_neighbor=1):
    #initialize weights
    random_W = np.random.randint(2, size=(shape_N, neurons_K)) - 1.
    num_neighbor_0 = int(shape_N // 2) - 1
    num_neighbor_1 = int(shape_N // 2) - 1
    for iterate in range(iteration):
        #Randomly select a training vector xi from X
        random_x = np.random.randint(len(word_vectors_X))

        #a very simple implementation of a SOM as described in the paper.
        w_xi = ((word_vectors_X[random_x, :].reshape(random_W.shape[0],1) - random_W))
        index = (np.unravel_index(np.argmin(w_xi), w_xi.shape))

        index_0_left = int(index[0] - num_neighbor_0)
        index_0_right = int(index[0] + num_neighbor_0 + 1)

        index_1_left = int(index[1] - num_neighbor_1)
        index_1_right = int(index[1] + num_neighbor_1 + 1)
        
        random_W[index_0_left:index_0_right, index_1_left:index_1_right] += learning_rate_a * w_xi[index]

        learning_rate_a = learning_rate_a * np.exp(-iterate / iteration)
        num_neighbor_0  =  (num_neighbor_0) * np.exp(-iterate / iteration)
        num_neighbor_1  =  (num_neighbor_1) * np.exp(-iterate / iteration)
    return random_W

if __name__ == "__main__":
    corpus = [["I cant get no"],["satisfaction"]]
    corpus = (remove_stopwords_from_corpus(corpus))
    large_object = (large_object_for_counts_vectors_clusters(corpus))

    SOM = (Kohonen_SOM(large_object["word_vectors"], large_object["word_vectors"].shape[1]))

    print("SOM, aka just the weight matrix SOM produces \n\n", SOM)
    #Created the SOM, was unclear about the relevance of the Document via search. That wasn't discussed
    #very clearly in the paper.    
