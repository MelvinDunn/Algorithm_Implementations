"""
CBOW implementation is used here.
"""

import numpy as np


def get_data(filepath):
    f = open(filepath, 'r')
    message = f.read()
    return message

def clean_text(corpus):
    return corpus.lower().split("\n")


def create_vocabulary(corpus):
    #set will just output whatever order so it has to be sorted.
    return sorted(list(set(corpus)))


if __name__ == "__main__":
    cleaned_corpus = (clean_text(get_data('data/steely_dan_deacon_blues_lyrics.txt')))
    print(cleaned_corpus)
    
    