"""
CBOW implementation is used here.
"""

import numpy as np

def get_data(filepath):
    f = open(filepath, 'r')
    message = f.read()
    return message

def clean_text(corpus):
    return corpus.lower().split()
 
def create_vocabulary(corpus):
    #set will just output whatever order so it has to be sorted.
    return sorted(list(set(corpus)))

def init_input_one_hot_layer(doc, word):
    zeros = np.zeros((len(doc),1))
    index_word = np.where(np.asarray(doc) == word)[0]
    zeros[index_word] += 1
    return zeros

def create_input_and_output_words(doc, window=1):
    inputs = []
    outputs = []
    window_range = list(range(-window, window+1))
    window_range.remove(0)
    item = 0
    for key, value in enumerate(doc):
        for j in window_range:
            if key == 0 and j < key:
                pass
            else:
                try:
                    input_item = doc[key]
                    output_item = doc[key+j]
                    inputs.append(input_item)
                    outputs.append(output_item)
                except IndexError:
                    pass
    return inputs,outputs

def create_one_hots(doc, io_words, input=True):

    if input == True:
        words = io_words[0]
    else:
        words = io_words[1]
    empty = np.zeros((len(words), len(doc)))
    for key,value in enumerate(words):
        empty[key,:] += (np.asarray(init_input_one_hot_layer(doc,value)).T.reshape(len(doc)))
    return empty

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def cbow(vocab, word):
    pass

def linear_neuron(x):
    return np.log(1+np.exp(x))

def init_hidden_layer(vocab, n_features=300):
    return np.zeros((len(vocab), n_features))

def shallow_network(inputs, output, epoch, hiddenlayer_neurons):
    #neural network portion is from https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/
    lr=0.1
    inputlayer_neurons = inputs.shape[0]
    output_neurons = (output.shape[0])
    #initialize neurons
    wh = np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
    bh = np.random.uniform(size=(1,hiddenlayer_neurons))
    wout = np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
    bout = np.random.uniform(size=(1,output_neurons))
    for i in range(epoch):
        hidden_layer_input1 = np.dot(inputs, wh)+bh
        hidden_layer_activations = sigmoid(hidden_layer_input1)
        #dot product of your activations and your weights plus biases gets your
        #inputs for the output layer
        output_layer_input = np.dot(hidden_layer_activations , wout) + bout
        #sigmoid transform the output layer
        y_hat_output = sigmoid(output_layer_input)
        #comput the error
        error = output - y_hat_output
        #comput the slope
        slope_output_layer = derivatives_sigmoid(y_hat_output)
        slope_hidden_layer = derivatives_sigmoid(hidden_layer_activations)
        delta_output = error * slope_output_layer * lr
        error_at_hidden_layer = np.dot(delta_output, wout.T)
        #compute the delta at the hidden layer.
        delta_hidden_layer = error_at_hidden_layer * slope_hidden_layer
        
        #update your weights and biases
        wout += np.dot(hidden_layer_activations.T, delta_output) * lr
        wh += np.dot(inputs.T.reshape(inputs.shape[0],1), delta_hidden_layer)*lr
        
        bh += np.sum(delta_hidden_layer, axis=0)*lr
        bout += np.sum(delta_output, axis=0)*lr
    return y_hat_output

def shallow_network_iter(inputs, output, epoch=2000, hiddenlayer_neurons=10):
    empty = np.zeros(inputs.shape)
    for i in range(inputs.shape[0]):
        empty[i,:] += (shallow_network(inputs[i,:],outputs[i,:],epoch, hiddenlayer_neurons)).reshape(inputs.shape[1])
    return empty

def derivatives_sigmoid(x):
    return x * (1. - x)

def cosine_sim(a,b):
    return np.dot(a,b) / np.dot(a.dot(a), b.dot(b))

def cosine_sim_two_words(word1, word2, yhat):
    return cosine_sim(y_hat[:,cleaned_corpus.index(word1)], y_hat[:,cleaned_corpus.index(word2)])


if __name__ == "__main__":
    cleaned_corpus = (clean_text(get_data('data/steely_dan_deacon_blues_lyrics.txt')))[:100]
    print(cleaned_corpus)
    io_words = (create_input_and_output_words(cleaned_corpus))
    inputs = create_one_hots(cleaned_corpus, io_words)
    outputs = create_one_hots(cleaned_corpus, io_words, input=False)
    lookup = np.asarray([cleaned_corpus for i in range(inputs.shape[0])])
    y_hat = ((shallow_network_iter(inputs,outputs)))
    #word2vec becomes relevant as the size of your corpus grows. Using only 100 words yields poor results.
    print(cosine_sim_two_words("this", "this", y_hat))
    
    