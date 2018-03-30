import numpy as np
from sklearn.datasets import load_iris
import pandas
import matplotlib.pyplot as plt

def linear_regression_main(data, y, intercept=True, iteration=100, alpha=0.001):
    #dot product of moore pseudoinverse and y variable yields coefficients.
    #moore psuedoinverse is a method of solving linear equations. In this case, least squares.
    X_0_intercept = np.ones((data.shape[0], 1))
    if intercept == True:
        data = np.hstack((X_0_intercept, data))
    else:
        pass
    rss_list = []
    #coeffs
    # be careful because this normal equation / moore penrose is 0(n^3)
    coeffs = np.matmul(np.matmul(np.linalg.inv(np.matmul(data.T, data)), data.T), y)
    #gradient descent iteration through each coefficient.
    for i in range(iteration):
        for i in range(len(coeffs)):
            coeffs[i] = (cost_function_gradient_descent(data, y, coeffs, data[:,i], coeffs[i], alpha))
            rss = linreg(data, y, coeffs)[2]
            print("New RSS for Lin Reg is {}".format(rss))
            rss_list.append(rss)
    return linreg(data, y, coeffs), rss_list

def linreg(data, y, coeffs):
    fit = np.sum(data * coeffs, axis = 1)
    residuals = fit - y
    return coeffs, residuals, rss(residuals)

def rss(residuals):
    return np.sum(residuals) ** 2


def cost_function_gradient_descent(data, y, coeffs, feature_j, coefficient_j, alpha):
    return coefficient_j - (alpha * (np.sum( (1 / float(data.shape[0])) * (data.dot(coeffs) - y) * feature_j)))

if __name__ == "__main__":
    data = (load_iris()["data"])
    target = (load_iris()["target"])
    rss = linear_regression_main(data, target)[1]
    pandas.DataFrame({"rss":rss}).plot()
    plt.show()