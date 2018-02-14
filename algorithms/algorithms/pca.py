"""
My implementation of PCA, just for my own edification.
Most implementations use the U matrix in SVD instead of the covariance matrix,
And the results are similar, but not the same. If you used scikit to compare it would be different.

Implemented from reading:
http://people.cs.pitt.edu/~iyad/PCA.pdf

Checked it with this, implemented scaling with this:
http://sebastianraschka.com/Articles/2015_pca_in_3_steps.html
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as sklearn_pca
from sklearn.preprocessing import StandardScaler


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

def the_ol_dot_product(data, eigs):
    return data.dot(eigs)

def PCA(data, n_components):
    data = StandardScaler().fit_transform(data)
    new_data = (calculate_covariance(subtract_by_mean(data)))
    eigens = compute_eigen(new_data)
    top_eigs = get_the_top_eigenvalues(eigens, n_components)
    pca = the_ol_dot_product(data, top_eigs)
    return pca

if __name__ == "__main__":
    data = (load_iris()["data"])
    print(PCA(data, 2))