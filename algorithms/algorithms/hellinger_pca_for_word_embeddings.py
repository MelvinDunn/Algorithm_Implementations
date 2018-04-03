#this is fine.
import numpy as np
import pandas as pd
import re

from sklearn.preprocessing import StandardScaler


MAX_ITEM = 16

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


#this is fine.
def remove_stopwords(words):
    """
    Only allows alphanumeric characters, cleans the stopwords and splits them, respectively.
    """
    words = re.sub(r'[^A-Za-z]', ' ', words)
    words = words.lower().split()
    return [word for word in words if word not in stopwords()]


#also fine.
def remove_stopwords_from_corpus(corpus):
    return [[" ".join(remove_stopwords(words[0]))] for words in corpus]
    

#fine
def get_data(filepath):
    f = open(filepath, 'r')
    message = f.read()
    return message

#this is fine.
def clean_text(corpus):
    return corpus.lower().split()


#create vocabulary
def create_vocabulary(corpus):
    #set will just output whatever order so it has to be sorted.
    return sorted(list(set(corpus)))


#this is fine.
def co_occurance_matrix(vocab):
    zeros = np.zeros((len(vocab),len(vocab)))
    return pd.DataFrame(zeros, index=vocab, columns=vocab)


def hellinger_distance(prob_dist_a, prob_dist_b):
    """
    takes in two probability distributions and computes their distance
    """
    p = np.sqrt(prob_dist_a)
    q = np.sqrt(prob_dist_b)
    return (-1 / np.sqrt(2)) * np.sqrt(np.sum(np.sum((p - q) ** 2)))


def count_words_add_to_matrix(co_occur, doc, window=1):
    window_range = list(range(-window, window+1))
    window_range.remove(0)
    for key, value in enumerate(doc):
        for j in window_range:
            if key == 0 and j < key:
                pass
            else:
                try:
                    #j is going to be your decay offset
                    co_occur.loc[doc[key], doc[key+j]] += 1 / abs(j)
                except IndexError:
                    pass
    Xi = np.sum(co_occur, axis=1)
    return co_occur / Xi


def pca_reconstruction(PCA_scores_matrix, eigenvectors, mean_vector):
    return np.dot(PCA_scores_matrix, eigenvectors.T) + mean_vector.reshape(mean_vector.shape[0],1)


def subtract_by_mean(data):
    data = StandardScaler().fit_transform(data)
    mean_data = np.zeros(data.shape)
    for i in range(data.shape[1]):
        mean_data[:, i] = data[:, i] - np.mean(data[:, i])
    return mean_data


def calculate_covariance(data):
    return np.cov(data, rowvar=False)

    
def compute_eigen(data):
    return np.linalg.eig(data)


def get_the_top_eigenvectors(eigens, n_components):
    #get the top eigenvalues
    #return the top eigenvectors corresponding to those eigenvalues.
    principal_eigenvalues_index = (eigens[0].argsort()[::-1][:n_components]).astype("int")
    eigenvectors = eigens[1][:,principal_eigenvalues_index].astype("float64")
    top_eigs = np.zeros((eigens[1].shape[1], principal_eigenvalues_index.shape[0]))
    for i in principal_eigenvalues_index:
        top_eigs[:, i] += eigenvectors[:, i]
    return top_eigs


def PCA(data, n_components):
    data = StandardScaler().fit_transform(data)
    new_data = (calculate_covariance(subtract_by_mean(data)))
    eigens = compute_eigen(new_data)
    top_eigs = get_the_top_eigenvectors(eigens, n_components)
    pca = data.dot(top_eigs)
    return pca, top_eigs, np.mean(data, axis=1)


def hellinger_pca(corpus, vocab):
    co_occur = co_occurance_matrix(vocab)
    co_occur = count_words_add_to_matrix(co_occur, corpus, window=2)
    best_reconstruction_matrix = {}
    best_reconstruction_matrix["n_components"] = []
    best_reconstruction_matrix["hellinger_distance_metric"] = []

    for i in range(1, MAX_ITEM):
        best_reconstruction_matrix["n_components"].append(i)
        temp_pca, eigenvectors, mean_vector = PCA(co_occur, i)
        best_reconstruction_matrix["hellinger_distance_metric"].append(\
            hellinger_distance(co_occur, pca_reconstruction(temp_pca, eigenvectors, mean_vector)))
    
    del temp_pca
    del eigenvectors
    del mean_vector

    best_reconstruction_matrix = pd.DataFrame(best_reconstruction_matrix)

    best_index = best_reconstruction_matrix.hellinger_distance_metric.argmin()

    hellinger_n_component = best_reconstruction_matrix.n_components.iloc[best_index]
    #best_pca_component = max(best_reconstruction_matrix, best_reconstruction_matrix.get)best

    hellinger_pca = PCA(co_occur, hellinger_n_component)[0]
    
    return hellinger_pca


def lookup_table(lookup_word, W, vocab):
    """
    assumes the vocab is in the same order as the weights matrix

    hellinger pca is the same thing as a weights matrix in this case.
    """
    lookup = vocab.index(lookup_word)
    return W[lookup,:]

if __name__ == "__main__":
    cleaned_corpus = (remove_stopwords(get_data('data/steely_dan_deacon_blues_lyrics.txt')))
    #It can just be continuous text since it's not using Documents.
    vocabulary = create_vocabulary(cleaned_corpus)
    hellinger_pca = (hellinger_pca(cleaned_corpus, vocabulary))
    print("The weights matrix vector for Alabama is \n", lookup_table("alabama", hellinger_pca, vocabulary))