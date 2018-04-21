"""
Not really an algorithm, just an experiment.

http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/

Kullback leibler doesn't really represent the relatedness very well.

Not really worth publishing. Just like an "oh I guess that's neat"
"""

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
import numpy as np
import re
from sklearn.cluster import KMeans
from gensim.scripts.glove2word2vec import glove2word2vec
import logging
import os
import pandas as pd

PATH = os.path.dirname(os.path.abspath(__file__))

def get_pretrained_word_vectors():
    glove_file = datapath(PATH + '/data/glove.6B.50d.txt')
    tmp_file = get_tmpfile("test_word2vec.txt")
    glove2word2vec(glove_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)
    word_vectors = model.wv
    return word_vectors


def kullback_leibler_divergence(pq, pd):
    """
    computes kullback leibler.
    """
    return np.sum(np.nan_to_num(pq * np.log(pq / pd)))

def kullback_main(w1, w2):
    p1 = word_vectors.wv.get_vector(w1)
    p2 = word_vectors.wv.get_vector(w2)
    return kullback_leibler_divergence(p1, p2)

if __name__ == "__main__":
    word_vectors = get_pretrained_word_vectors()
    data = pd.read_csv(PATH + '/data/wordsim_combined.csv')
    items = []
    word1 = []
    word2 = []
    kullback = []
    for i in range(data.shape[0]):
        try:
            #cosine
            items.append(word_vectors.wv.similarity(data["Word 1"].iloc[i], data["Word 2"].iloc[i]))
            word1.append(data["Word 1"].iloc[i])
            word2.append(data["Word 2"].iloc[i])
            kullback.append(kullback_main(data["Word 1"].iloc[i], data["Word 2"].iloc[i]))
        except Exception:
            pass

    #items = np.abs(np.asarray(items) - np.asarray(data["Human (mean)"]))
    pd.DataFrame({"word1": word1, "word2": word2 ,"glove6B50_cosine": items, "glove6B50_kullback": kullback}).to_csv(PATH + '/results/distance_metric_eval.csv')   
