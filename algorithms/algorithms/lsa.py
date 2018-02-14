import numpy as np
import pandas as pd

np.random.seed(3)
n_components = 2

def document_term_matrix(documents):
	unique_wordies = unique_words(documents)
	#just cleanup
	pre_matrix_list = []
	for docs in documents:
		pre_matrix_list.append([docs[0].count(i) for i in unique_wordies])
	return pd.DataFrame(pre_matrix_list, columns = unique_wordies)

def unique_words(documents):
	#combine the strings with list iteration
	docs_combined = [''.join(i) for i in documents]
	#get the unique items in the strings via a set
	unique_words = sorted(list(set((" ".join(docs_combined)).split())))
	return unique_words

def covariances(X):
	return X.dot(X.T)

def SVD(matrix):
	#sigma is the singular values.
	U, Sigma, V_t = np.linalg.svd(matrix, full_matrices=True)
	return U, Sigma, V_t

#LSI is the same thing as LSA.
def LSI(document_term_matrix, k=2):
	if k > document_term_matrix.shape[1]:
		print("k or number of componenets cannot be larger than columns")
	U, Sigma, V_t = (SVD((document_term_matrix)))
	Sigma = np.diag(Sigma)
	lsi = document_term_matrix.T.dot(U[:,:k]).dot(Sigma[:k,:k])
	return lsi

if __name__ == "__main__":
	#a list of documents is a corpus, fyi
	documents = [
		["I like to move it move it"],
		["You like to move it move it"],
		["We like to move it"]
	]
	document_term_matrix = (np.asarray(document_term_matrix(documents)))
	print("LSI value for {} components is: \n {}".format(n_components, LSI(document_term_matrix, k=n_components)))