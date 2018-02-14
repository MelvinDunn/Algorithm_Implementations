import numpy as np
import pandas as pd

def document_term_matrix(documents):
	unique_wordies = unique_words(documents)
	#just cleanup
	pre_matrix_list = []
	for docs in documents:
		#pretty slow implementation by appending to a list,
		#but for pedagogy I'm going to let it slide.
		pre_matrix_list.append([docs[0].count(i) for i in unique_wordies])
	return pd.DataFrame(pre_matrix_list, columns = unique_wordies)

def unique_words(documents):
	#combine the strings with list iteration
	docs_combined = [''.join(i) for i in documents]
	#get the unique items in the strings via a set
	unique_words = list(set((" ".join(docs_combined)).split()))
	return unique_words

def term_frequency(document_term_matrix):
	sum_of_words_in_document = document_term_matrix.sum(axis=1).values.T
	return (document_term_matrix.T / sum_of_words_in_document).T

def inverse_document_frequency(document_term_matrix):
	total_number_of_documents = document_term_matrix.shape[0]
	number_of_documents_with_term_t = document_term_matrix.copy()
	number_of_documents_with_term_t[document_term_matrix > 0] = 1
	number_of_documents_with_term_t = number_of_documents_with_term_t.sum(axis=0)
	return np.log(total_number_of_documents / number_of_documents_with_term_t)

def tfidf(document_term_matrix):
	return term_frequency(document_term_matrix).dot(inverse_document_frequency(document_term_matrix))

if __name__ == "__main__":
	#a list of documents is a corpus, fyi
	documents = [
		["I like to move it move it"],
		["You like to move it move it"],
		["We like to move it"]
	]
	document_term_matrix = (document_term_matrix(documents))
	print("The TFIDF score of the document is: \n {}".format(tfidf(document_term_matrix)))
