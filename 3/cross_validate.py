from typing import Tuple
import numpy as np
from base_estimator import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data. Has functions: fit, predict, loss

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    n = X.shape[0]
    m = X.shape[1]
    train_score = 0
    validation_score = 0
    fold_sizes = np.full(cv, n // cv)
    fold_sizes[:n % cv] += 1
    cur = 0
    for i in range(cv):
        start, end = cur, cur + fold_sizes[i]
        X_train = np.concatenate((X[:start], X[end:]))
        y_train = np.concatenate((y[:start], y[end:]))
        X_test = X[start:end]
        y_test = y[start:end]
        estimator.fit(X_train, y_train)
        train_score += estimator.loss(X_train, y_train)
        validation_score += estimator.loss(X_test, y_test)
    return train_score / cv, validation_score / cv

