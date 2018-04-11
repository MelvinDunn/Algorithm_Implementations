import numpy as np

def svd(A):
	"""
	take in a matrix, A, and compute the SVD.
	"""
	#step 1 - compute A.T * A
	#A.T
	ATA = np.matmul(A.T, A)
	#compute the eigenvalues
	eigenvalues, eigenvectors = np.linalg.eig(ATA)
	S = np.diag(np.sort(eigenvalues)[::-1])
	S_inv = np.linalg.inv(S)
	V_t = eigenvectors.T
	U = np.matmul(np.matmul(A,eigenvectors),S_inv)
	return U, S, V_t


if __name__ == "__main__":
	A = np.array([[4, 0],[3,-5]])
	U, S, V_t = svd(A)
	#recombined SVD
	A_hat = np.matmul(np.matmul(U,S),V_t)
	#supposed to be true.
	print("Recombining USV_t to form A is {}".format((A - A_hat) < 0.000000001))
