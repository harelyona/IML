import numpy as np
import plotly.graph_objects as go
from sklearn import datasets
from cross_validate import cross_validate
from estimators import RidgeRegression, Lasso, LinearRegression


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
    # Question 1 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 2 - Perform Cross Validation for different values of the regularization parameter for Ridge and
    # Lasso regressions
    raise NotImplementedError()

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    raise NotImplementedError()
