import numpy as np
from sklearn.datasets import load_iris
"""
ridge_value is lambda in Hastie text.

This monotone decreasing function of λ is the effective degrees of freedom
of the ridge regression fit. Usually in a linear-regression fit with p variables,
the degrees-of-freedom of the fit is p, the number of free parameters. The
idea is that although all p coefficients in a ridge fit will be non-zero, they
are fit in a restricted fashion controlled by λ. Note that df(λ) = p when
λ = 0 (no regularization) and df(λ) → 0 as λ → ∞. Of course there
is always an additional one degree of freedom for the intercept, which was
removed apriori. This definition is motivated in more detail in Section 3.4.4
and Sections 7.4–7.6. In Figure 3.7 the minimum occurs at df(λ) = 5.0.
Table 3.3 shows that ridge regression reduces the test error of the full least
squares estimates by a small amount.

http://statsmaths.github.io/stat612/lectures/lec17/lecture17.pdf :

Ordinary least squares and ridge regression have what are called
analytic solutions, we can write down an explicit formula for what
the estimators are.
"""

def main(X, y, ridge_value=0.1):
    #Taken from figure 3.47 of elements of statistical learning by Hastie.
	lambda_I = np.identity(X.shape[1]) * ridge_value
	return np.dot(np.dot(np.linalg.inv(np.dot(data.T, data)+lambda_I), data.T), y)

def rss(residuals):
    return np.sum(residuals) ** 2

if __name__ == "__main__":
    data = (load_iris()["data"])
    original_data = data.copy()
    y = (load_iris()["target"])
    intercept = np.ones((data.shape[0],1))
    data = np.column_stack((intercept, data))

    #shrinkage doesn't work without an intercept. lesson learned, it would seem.
    print(main(data,y,0.0001))
    print(main(data,y,0.001))
    print(main(data,y,0.1))
    print(main(data,y,0.7))
    print(main(data,y,0.8))
    print(main(data,y,0.95))
