import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def lda(data, target, n_components=2):
    within = (within_class_scatter(data,target))
    between = (between_class_scatter(data,target))
    eigenvalues, eigenvectors = (generalized_eigenvalue(within,between))
    top_eigenvalues_index = (eigenvalues.argsort()[::-1][:n_components])
    W = eigenvectors[:, top_eigenvalues_index]
    return data.dot(W)

def get_means_for_each_class(data, target, unique_targets):
    return [np.mean(data[target==i],axis=0) for i in unique_targets]

def unique_targets(target):
    return sorted(list(set(target.tolist())))

def within_class_scatter(data, target):
    unique_targs = unique_targets(target)
    class_means = get_means_for_each_class(data, target, unique_targs)
    empty = np.zeros((data.shape[1],data.shape[1]))
    for unique in unique_targs:
        group = (data[target == unique])
        group_minus_means = (group - class_means[unique]).T
        empty += (group_minus_means.dot(group_minus_means.T))
    return empty

def between_class_scatter(data,target):
    #unique targets are the classes
    unique_targs = unique_targets(target)
    class_means = get_means_for_each_class(data, target, unique_targs)
    empty = np.zeros((data.shape[1],data.shape[1]))
    overall_mean = np.mean(data, axis=0)
    for unique in unique_targs:
        sample_size_of_class = target[target == unique].shape[0]
        item = (class_means[unique] - overall_mean).T.reshape(data.shape[1],1)
        empty += sample_size_of_class * (item.dot(item.T))
    return empty

def generalized_eigenvalue(within_class_scatter,between_class_scatter):
    return np.linalg.eig(np.linalg.inv(within_class_scatter).dot(between_class_scatter))

if __name__ == "__main__":
    data = (load_iris()["data"])
    target = (load_iris()["target"])
    #split data
    #X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)
    print(lda(data, target))
    
    
    
    