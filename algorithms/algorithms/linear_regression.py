import numpy as np
from sklearn.datasets import load_iris

def linear_regression_main(data, y, n_iter = 1):
    #dot product of moore pseudoinverse and y variable yields coefficients.
    #moore psuedoinverse is a method of solving linear equations. In this case, least squares.
    coeffs = np.dot(np.dot(np.linalg.inv(np.dot(data.T, data)), data.T), y)
    return linreg(data, y, coeffs)

def linreg(data, y, coeffs):
    fit = np.sum(data * coeffs, axis = 1)
    residuals = fit - y
    return coeffs, residuals, rss(residuals)

def rss(residuals):
    return np.sum(residuals) ** 2

if __name__ == "__main__":
    data = (load_iris()["data"])
    target = (load_iris()["target"])
    print(linear_regression_main(data, target))