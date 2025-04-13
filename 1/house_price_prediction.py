import os
from linear_regression import *
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

random.seed(1)


def id_and_dates(X):
    """drop ids and split dates"""
    # Id is an irrelevant feature
    # the condition  "id" in X.columns is True
    if "id" in X.columns:
        X = X.drop("id", axis=1)  # So why this line fails with KeyError: "['id'] not found in axis"????
    # Replace date with days counter from May 1st 2014
    if "date" in X.columns:
        X["days since May 1st 2014"] = (pd.to_datetime(X["date"]) - pd.Timestamp("2014-05-01")).dt.days
        X.drop('date', axis=1, inplace=True)
    return X


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
    idx_to_drop = []
    for idx in X.index:
        time_built = pd.to_datetime(X["yr_built"][idx])
        if time_built > pd.to_datetime(X["yr_renovated"][idx]) or time_built > pd.to_datetime(X["date"][idx]):
            idx_to_drop.append(idx)
    X.drop(idx_to_drop, inplace=True)
    y.drop(idx_to_drop, inplace=True)
    X = id_and_dates(X)

    # Get rid of nan samples
    nan_indices_X = X.isna().any(axis=1)
    nan_indices_y = y.isna()
    combined_nan_indices = nan_indices_X | nan_indices_y

    # Convert all elements to numeric values
    for col in X.columns:
        X[col] = pd.to_numeric(X[col],errors="raise")
    y = pd.to_numeric(y, errors="raise")
    X, y = X[~combined_nan_indices], y[~combined_nan_indices]

    return X, y


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
    for idx in X.index:
        time_built = pd.to_datetime(X["yr_built"][idx])
        if time_built > pd.to_datetime(X["yr_renovated"][idx]):
            X["yr_built"][idx] = X[["yr_built"]].mean()
            X["yr_renovated"][idx] = X[["yr_renovated"]].mean()
        elif time_built > pd.to_datetime(X["date"][idx]):
            X["yr_built"][idx] = X[["yr_built"]].mean()
            X["date"] = X[["date"]].mean()

    X = id_and_dates(X)
    # Replace every nan element to the mean value of its column
    for feature in X:
        if feature == "date":
            continue
        mean = X[feature].mean()
        X[feature] = X[feature].fillna(mean)
    for col in X:
        X[col] = pd.to_numeric(X[col], errors="raise")
    return X


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
    X, response = preprocess_train(X, y)
    pearson_correlations = {}
    response_std = response.std()
    for col in X.columns:
        feature = X[col]
        feature_std = feature.std()
        cov = np.cov(feature, response)[0][1]

        # Check for zero standard deviation to avoid division by zero
        if feature_std > 0 and response_std > 0:
            pearson = cov / (feature_std * response_std)
            pearson_correlations[col] = pearson
        else:
            pearson_correlations[col] = 0

        # Create plot
        plt.title(f"house price vs {col}")
        plt.scatter(feature, response)
        plt.xlabel(col)
        plt.ylabel("price")
        plt.savefig(f"{output_path}{os.sep}{col}.png", format="png")
        plt.clf()
    return pd.Series(pearson_correlations)


if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    df, y = df.drop("price", axis=1), df.price
    output_path = "plots"

    # Question 2 - split train test
    m, n = df.shape
    test_data_size = int(m * 0.25)
    training_data_size = m - test_data_size
    test_start = random.randint(0, int(m - test_data_size))
    test_sample = df.iloc[test_start:test_start + test_data_size]
    test_response = y.iloc[test_start:test_start + test_data_size]
    training_samples = df.drop(test_sample.index)
    response = y.drop(test_sample.index)
    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    test_samples = preprocess_test(test_sample)
    test_samples = test_samples.astype(np.float64)
    percentages = range(10, 15)
    mean_loss = []
    std_loss = []
    for p in percentages:
        print(p)
        losses = []
        for _ in range(10):
            number_of_samples = int(training_data_size * p / 100)
            print(1)
            pre_processed_X = training_samples.sample(number_of_samples)
            print(2)
            y = response[pre_processed_X.index]
            print(3)
            X, y = preprocess_train(pre_processed_X, y)
            print(4)
            X = X.astype(np.float64)

            y = y.astype(np.float64)
            X = X.to_numpy()
            y = y.to_numpy()
            fit = LinearRegression(True)
            fit.fit(X, y)
            losses.append(fit.loss(test_samples, test_response))
        mean_loss.append(np.mean(losses))
        std_loss.append(np.std(losses))
    plt.xlabel("training sample percentage")
    plt.ylabel("loss")
    plt.errorbar(percentages, mean_loss, yerr=std_loss, label="mean loss", fmt='o')
    plt.savefig(output_path + os.sep + "mean_loss.png", format="png")
    plt.show()







