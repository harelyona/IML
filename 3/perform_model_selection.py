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
    data = datasets.load_diabetes()
    X = data.data
    y = data.target
    # Question 2 - Perform Cross Validation for different values of the regularization parameter for Ridge and
    # Lasso regressions
    lasso_train_errors = np.zeros(n_evaluations)
    lasso_validation_errors = np.zeros(n_evaluations)
    ridge_train_errors = np.zeros(n_evaluations)
    ridge_validation_errors = np.zeros(n_evaluations)
    ridge_regulazations = np.linspace(1, 5, num=n_evaluations, dtype=float)
    lasso_regulazations = np.linspace(.001, 2, num=n_evaluations, dtype=float)
    for i in range(n_evaluations):
        lasso = Lasso(float(lasso_regulazations[i]))
        ridge = RidgeRegression(float(ridge_regulazations[i]))
        lasso_train_errors[i], lasso_validation_errors[i] = cross_validate(lasso, X, y)
        ridge_train_errors[i], ridge_validation_errors[i] = cross_validate(ridge, X, y)

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
