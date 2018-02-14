"""
1) compute sigmoid function
2) use weight initialization from linear regression - the pseudo inverse used
for solving the matrix 
"""
import numpy as np

def sigmoid(x):
    return (1 / (1. + np.exp(-x)))

def moore_penrose_pseudoinverse(data, y):
    #this will break if you have two columns that are the same.
    #breaks with collinearity. - Particularly the inverse portion.
    return np.dot(np.dot(np.linalg.inv(np.dot(data.T, data)), data.T), y)

def estimate_coeffs(data, y):
    #returns maximum likelihood for estimating coefficients, takes in moore penrose as base coefficients.
    #https://stats.stackexchange.com/questions/229014/matrix-notation-for-logistic-regression
    #you can approximate the coefficient using this MLE
    #Many people just prefer to use gradient descent for the entire thing
    #but that's kind of garbage. If you used moore-penrose by itself it's not that reliable.
    #this needs to be tested with more data to see if it's relevant.
    moore_pen_coeffs = (moore_penrose_pseudoinverse(data, y).T * data)
    return np.min(np.log(1. + np.exp(-moore_pen_coeffs))+\
    ((1.-y)*np.log(1. + np.exp(moore_pen_coeffs))),axis=0)

def log_reg(data,y,num_iterations=1000, learning_rate=5e-5, add_intercept=True):
    if add_intercept:
        intercept = np.ones((data.shape[0], 1))
        data = np.hstack((intercept, data))
    #initialize coefficients
    coeffs = estimate_coeffs(data, y)
    preds = np.round(sigmoid(np.dot(data, coeffs)))
    preds = (preds.reshape(y.shape))
    print('Accuracy before gradient descent: {}'.format((preds == y).sum() / y.shape[0]))
    #gradient portion stolen from https://beckernick.github.io/logistic-regression-from-scratch/
    #basically takes the 
    for i in range(num_iterations):
        preds = log_reg_scores(data, y, coeffs).reshape(y.shape)
        diff = (y - preds)
        gradient = (np.dot(data.T, diff)).reshape(coeffs.shape)
        coeffs += learning_rate * gradient
    preds = np.round(sigmoid(np.dot(data, coeffs)))
    preds = (preds.reshape(y.shape))
    print('Accuracy after gradient descent: {}'.format((preds == y).sum() / y.shape[0]))
    return 'Coefficients: {}'.format(coeffs)

def log_reg_scores(data,y, coeffs):
    X = np.dot(data , coeffs)
    return sigmoid(X)

if __name__ == "__main__":
    np.random.seed(12)
    num_observations = 5000

    #Used data from from https://beckernick.github.io/logistic-regression-from-scratch/ to check my work 
    x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

    data = np.vstack((x1, x2)).astype(np.float32)
    y = np.hstack((np.zeros(num_observations),
                                  np.ones(num_observations))).reshape(num_observations*2,1)
    print(log_reg(data,y))