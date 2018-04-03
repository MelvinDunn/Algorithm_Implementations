"""
From Andrew Ng's anamoly detection algorithm explanation from Coursera

This would see if the item falls as an outlier to the gaussian distribution.
"""

import numpy as np
from sklearn.datasets import load_iris
from math import pi

def anamoly_detection_univariate(X, epsilon=0.001):
    """
    Choose features you think would take on abnormally large values.
    
    Univariate scales better.
    """
    
    mu_vector = X.mean(axis = 0)
    sigma_vector = X.std(axis = 0)
    
    main_output = (1 / np.sqrt(2 * pi * sigma_vector)) \
    * np.exp(-((X - mu_vector) ** 2) / 2 * (sigma_vector ** 2))
    
    return (main_output < epsilon)


def anamoly_detection_multivariate(X, epsilon=0.01):
    """
    Choose features you think would take on abnormally large values.
    
    Automatically captures correlations between features, so you don't
    have to randomly compute correlations of your main output.
    """
    
    mu_vector = X.mean(axis = 0)
    sigma_vector = X.std(axis = 0)
    sigma_vector = np.diag(sigma_vector)
    first_portion_equation = (1 / ((2 * pi) ** (X.shape[1] / 2.)))
    second_portion_equation = (-1/2.)*(X - mu_vector)
    third_portion_equation = np.matmul((X-mu_vector),np.linalg.inv(sigma_vector))
    main_output =  first_portion_equation * np.exp(second_portion_equation * third_portion_equation)
    main_output = (1 / ((2 * pi) ** (X.shape[1] / 2.))) * np.exp((-1/2.)*(X - mu_vector) * np.matmul((X-mu_vector),np.linalg.inv(sigma_vector)))
    
    return (main_output < epsilon)


if __name__ == "__main__":
    data = (load_iris()["data"])
    print(anamoly_detection_multivariate(data))
    print(anamoly_detection_univariate(data))