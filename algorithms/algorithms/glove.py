import numpy as np
import re
import pandas as pd
XMAX = 10
ALPHA = 3/4.
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
    

def get_data(filepath):
    f = open(filepath, 'r')
    message = f.read()
    return message
    
def clean_text(corpus):
    return corpus.lower().split()


def create_vocabulary(corpus):
    #set will just output whatever order so it has to be sorted.
    return sorted(list(set(corpus)))

def co_occurance_matrix(vocab):
    zeros = np.zeros((len(vocab),len(vocab)))
    return pd.DataFrame(zeros, index=vocab, columns=vocab)


def count_words_add_to_matrix(co_occur, doc, window=1):
    window_range = list(range(-window, window+1))
    window_range.remove(0)
    for key, value in enumerate(doc):
        print(value)
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

def cost_function_J(co_occur, main_word_vector, context_word_vector, bias_i, bias_j):
    return weighting_function_f(co_occur, alpha=ALPHA) * ((np.dot(main_word_vector.T, context_word_vector) \
            + bias_i  + bias_j - np.log(co_occur)) ** 2)


def cost_function_iter(co_occur, main_word_vector, context_word_vector, bias_i, bias_j):
    for i in range(len(co_occur.shape[0])):
        for j in range(len(co_occur.shape[0])):        
            #J = cost_function_J(co_occur, co_occur[i, j], co_occur.iloc[], bias_i, bias_j)
            pass
        

def weighting_function_f(X, x_max=XMAX):
    index = X < x_max
    X[X == True] /= x_max
    X[X == False] = 1.
    return X

def glove_main(corpus, vocab, x_max=XMAX, iteration=10, vector_size = 300):
    co_occur_matrix = co_occurance_matrix(vocab)
    co_occur_matrix = count_words_add_to_matrix(co_occur_matrix, corpus, window=2)
    #init stuff
    bias_i = np.random.normal(size=(vector_size,1))
    bias_j = np.random.normal(size=(vector_size,1))
    main_word_vector_matrix = np.zeros(size=(vector_size, len(vocab)))
    context_word_vector_matrix = main_word_vector_matrix.copy()
    
    return co_occur_matrix

if __name__ == "__main__":
    cleaned_corpus = (remove_stopwords(get_data('data/steely_dan_deacon_blues_lyrics.txt')))
    #It can just be continuous text since it's not using Documents.
    vocabulary = create_vocabulary(cleaned_corpus)
    print((glove_main(cleaned_corpus, vocabulary)))
    
    
    
    
