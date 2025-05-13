from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    x, y = datasets.load_diabetes(return_X_y=True)
    # train_X, train_y, test_X, test_y = x[:n_samples], y[:n_samples], x[n_samples:], y[n_samples:]
    n = x.shape[0]
    train_proportion = n_samples / n
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(x), pd.DataFrame(y), train_proportion=train_proportion)
    
    train_X = train_X.values
    test_X = test_X.values
    train_y = train_y.values
    test_y = test_y.values

    # Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ridge_alphas, lasso_alphas = np.linspace(1e-5, 5*1e-2, num=n_evaluations), np.linspace(.001, 2, num=n_evaluations)
    ridge_scores, lasso_scores = np.zeros((n_evaluations, 2)), np.zeros((n_evaluations, 2))
    for i, (ra, la) in enumerate(zip(*[ridge_alphas, lasso_alphas])):
        ridge_scores[i] = cross_validate(RidgeRegression(ra), train_X, train_y, mean_square_error)
        lasso_scores[i] = cross_validate(Lasso(la, max_iter=5000), train_X, train_y, mean_square_error)

    fig = make_subplots(1, 2, subplot_titles=[r"$\text{Ridge Regression}$", r"$\text{Lasso Regression}$"], shared_xaxes=True)\
        .update_layout(title=r"$\text{Train and Validation Errors (averaged over the k-folds)}$", width=750, height=300)\
        .update_xaxes(title=r"$\lambda\text{ - Regularization parameter}$")\
        .add_traces([go.Scatter(x=ridge_alphas, y=ridge_scores[:, 0], name="Ridge Train Error"),
                    go.Scatter(x=ridge_alphas, y=ridge_scores[:, 1], name="Ridge Validation Error"),
                    go.Scatter(x=lasso_alphas, y=lasso_scores[:, 0], name="Lasso Train Error"),
                    go.Scatter(x=lasso_alphas, y=lasso_scores[:, 1], name="Lasso Validation Error")],
                    rows=[1, 1, 1, 1],
                    cols=[1, 1, 2, 2])
    fig.write_image("ridge_lasso_train_validation_errors.png")

    # Compare best Ridge model, best Lasso model and Least Squares model
    reg_ridge = ridge_alphas[np.argmin(ridge_scores[:, 1])]
    reg_lasso = lasso_alphas[np.argmin(lasso_scores[:, 1])]
    print(f"# ------------------------------------------------------------------------- #\n"
          f"#    Regularized Regression Fitting - Selecting Regularization Parameter    #\n"
          f"# ------------------------------------------------------------------------- #\n"
          f"\tCV-optimal Ridge regularization parameter: {reg_ridge}\n"
          f"\tCV-optimal Lasso regularization parameter: {reg_lasso}\n"
          f"\n\n"
          f"Test errors of models:\n"
          f"\tLeast-Squares: {LinearRegression().fit(train_X, train_y).loss(test_X, test_y)}\n"
          f"\tRidge: {RidgeRegression(lam=reg_ridge).fit(train_X, train_y).loss(test_X, test_y)}\n"
          f"\tLasso: {mean_square_error(test_y,Lasso(alpha=reg_lasso).fit(train_X, train_y).predict(test_X))}\n"
          f"# ------------------------------------------------------------------------- #\n\n")

    



if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter(n_evaluations=500)
