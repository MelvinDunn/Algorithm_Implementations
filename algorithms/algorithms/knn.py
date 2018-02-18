import numpy as np
from sklearn.datasets import load_iris
from statistics import mode

def knn_classifier(X, y, k=10):
    empty = np.zeros(y.shape)
    for i in range(X.shape[0]):
        #basically fill this with an absurd number so argmin doesn't match on itself
        distances = np.abs(X[i,:] - X)
        distances[i] = 999999999999999
        arr = ((np.sum(distances, axis=1)))
        top_k_indices = (arr.argsort()[:k])
        #if the mode finds two equally occuring values you take the mean.
        #if you wanted to do the regression you would basically just take the mean below.
        try:
            empty[i] += mode(y[top_k_indices])
        except:
            empty[i] += np.mean(y[top_k_indices])
    return empty

def accuracy(y_hat, y):
    return "Accuracy for the knn classifier is {}".format(round(np.sum((y == y_hat)) / y.shape[0], 4))

if __name__ == "__main__":
    data = (load_iris()["data"])
    target = (load_iris()["target"])
    y_hat = (knn_classifier(data,target, k=1))
    y_hat2 = (knn_classifier(data,target, k=20))
    print(accuracy(y_hat, target))
    print(accuracy(y_hat2, target))    