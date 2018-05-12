import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as sklearn_pca
from sklearn.preprocessing import StandardScaler


def calculate_covariance(data):
    return np.cov(data, rowvar=False)


def Truncated_SVD(data, n_components):
	if n_components >= data.shape[1]:
		return "Number of components have to be less than the number of columns, or {}".format(data.shape[1])
    # don't subtract by the mean. I left out scaling as well. That's the difference between truncated SVD and PCA.
	Sigma_if_it_were = calculate_covariance((data))
	U,S,V = np.linalg.svd(Sigma_if_it_were)
	min_list = []
	#this is just to get an accurate k
	#there is a PCA reconstruction in the other file, which is anothber mothed for choosing k
	for i in range(data.shape[1]-1):
        #percentage of variance retained from reducing dimensions.
		SS = 1 - (np.sum(S[:i]) /  np.sum(S))
		min_list.append(SS)
	print((min_list))
	return data.dot(U[:n_components, :].T)

if __name__ == "__main__":
    data = (load_iris()["data"])
    print(Truncated_SVD(data, 2))