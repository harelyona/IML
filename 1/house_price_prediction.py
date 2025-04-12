from typing import NoReturn

import numpy as np
import pandas as pd
import random
import statistics
random.seed(1)
def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    pass


def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    return


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    pass


if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    df, y = df.drop("price", axis=1), df.price

    # Question 2 - split train test
    m, n = df.shape
    number_of_test_sampels = int(m*0.25)
    test_start = random.randint(0, int(m - number_of_test_sampels))
    test_sample = df.iloc[test_start:test_start + number_of_test_sampels]
    X = df.drop(test_sample.index)
    # Question 3 - preprocessing of housing prices train dataset

    # Question 4 - Feature evaluation of train dataset with respect to response
    def feature_evaluation(X: pd.DataFrame, response: pd.Series) -> pd.Series:
        pearson_correlations = {}

        response_std = statistics.stdev(response)

        for col in X.columns:
            feature = X[col]
            feature_std = statistics.stdev(feature)
            cov = statistics.covariance(feature, response_std)

            pearson = cov / (feature_std * response_std)
            pearson_correlations[col] = pearson
        return pd.Series(pearson_correlations)
    # Question 5 - preprocess the test data
    print(feature_evaluation(X, y))
    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

