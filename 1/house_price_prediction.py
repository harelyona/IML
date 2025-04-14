import os
from typing import Tuple

from linear_regression import *
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

random.seed(1)
ERROR_BAR_COLOR = "red"
CAP_SIZE = 6


def id_and_dates(X):
    """drop ids and split dates"""
    X_copy = X.copy()
    if "id" in X_copy.columns:
        X_copy = X_copy.drop("id", axis=1)

    # Replace date with days counter from May 1st 2014
    if "date" in X_copy.columns:
        X_copy["days since May 1st 2014"] = (pd.to_datetime(X_copy["date"]) - pd.Timestamp("2014-05-01")).dt.days
        X_copy.drop('date', axis=1, inplace=True)

    return X_copy


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
    X_processed = X.copy()
    y_processed = y.copy()
    X_processed['date'] = pd.to_datetime(X_processed['date'], errors='coerce')
    X_processed['yr_built_dt'] = pd.to_datetime(X_processed['yr_built'], format='%Y', errors='coerce')
    renovated_dt = pd.to_datetime(X_processed['yr_renovated'], format='%Y', errors='coerce')
    renovated_dt[X_processed['yr_renovated'].isin([0, pd.NA, None]) | renovated_dt.isna()] = pd.NaT
    X_processed['yr_renovated_dt'] = renovated_dt
    cond1 = ~X_processed['yr_renovated_dt'].isna() & (X_processed['yr_built_dt'] > X_processed['yr_renovated_dt'])
    cond2 = ~X_processed['date'].isna() & (
                X_processed['yr_built_dt'] > X_processed['date'])  # Added check for valid date
    invalid_date_logic_mask = cond1 | cond2 | X_processed['yr_built_dt'].isna() | X_processed['date'].isna()
    mask_to_keep = ~invalid_date_logic_mask
    X_processed = X_processed.loc[mask_to_keep]
    y_processed = y_processed.loc[mask_to_keep]
    X_processed = X_processed.drop(columns=['yr_built_dt', 'yr_renovated_dt'], errors='ignore')
    X_processed = id_and_dates(X_processed)
    for col in X_processed.columns:
        X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
    y_processed = pd.to_numeric(y_processed, errors='coerce')
    nan_mask_X = X_processed.isna().any(axis=1)
    nan_mask_y = y_processed.isna()
    combined_nan_mask = nan_mask_X | nan_mask_y
    X_final = X_processed.loc[~combined_nan_mask]
    y_final = y_processed.loc[~combined_nan_mask]
    return X_final, y_final


def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    """
    X_processed = X.copy()
    X_processed['date'] = pd.to_datetime(X_processed['date'], errors='coerce')
    X_processed['yr_built'] = pd.to_datetime(X_processed['yr_built'], format='%Y', errors='coerce')
    X_processed['yr_renovated'] = pd.to_datetime(X_processed['yr_renovated'], format='%Y', errors='coerce')
    mask = X_processed['yr_renovated'].isin([0, pd.NaT]) | X_processed['yr_renovated'].isna()
    X_processed.loc[mask, 'yr_renovated'] = pd.NaT
    mask_built_after_reno = (~X_processed['yr_renovated'].isna()) & (
                X_processed['yr_built'] > X_processed['yr_renovated'])
    X_processed.loc[mask_built_after_reno, 'yr_built'] = X_processed['yr_built'].mean()
    X_processed.loc[mask_built_after_reno, 'yr_renovated'] = X_processed['yr_renovated'].mean()
    mask_built_after_sale = (~X_processed['date'].isna()) & (X_processed['yr_built'] > X_processed['date'])
    X_processed.loc[mask_built_after_sale, 'yr_built'] = X_processed['yr_built'].mean()
    X_processed.loc[mask_built_after_sale, 'date'] = X_processed['date'].mean()
    X_processed['yr_built'] = X_processed['yr_built'].dt.year
    X_processed['yr_renovated'] = X_processed['yr_renovated'].dt.year
    X_processed['date'] = X_processed['date'].dt.strftime('%Y-%m-%d')
    X_processed = id_and_dates(X_processed)
    for feature in X_processed.columns:
        X_processed[feature] = pd.to_numeric(X_processed[feature], errors="coerce")
        X_processed[feature] = X_processed[feature].fillna(X_processed[feature].mean())
    return X_processed


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


def generate_sets_and_responses(data_frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    m, n = data_frame.shape
    test_data_size = int(m * 0.25)
    test_start = random.randint(0, int(m - test_data_size))
    test_samples = data_frame.iloc[test_start:test_start + test_data_size]
    test_response = y.iloc[test_start:test_start + test_data_size]
    training_samples = data_frame.drop(test_samples.index)
    training_response = y.drop(test_samples.index)
    return training_samples, training_response, test_samples, test_response


if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    df, y = df.drop("price", axis=1), df.price
    output_path = "plots"
    m, n = df.shape
    # Question 2 - split train test
    training_samples, response, test_samples, test_response = generate_sets_and_responses(df)
    training_data_size = m - test_samples.shape[0]

    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    test_samples = preprocess_test(test_samples)
    test_samples = test_samples.astype(np.float64)
    percentages = range(10, 101)
    mean_loss = []
    std_loss = []
    for p in percentages:
        losses = []
        for _ in range(10):
            number_of_samples = int(training_data_size * p / 100)
            pre_processed_X = training_samples.sample(number_of_samples)
            y = response[pre_processed_X.index]
            X, y = preprocess_train(pre_processed_X, y)
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
    plt.errorbar(percentages, mean_loss, yerr=std_loss, label="mean loss", fmt='o', ecolor=ERROR_BAR_COLOR, capsize=CAP_SIZE)
    plt.savefig(output_path + os.sep + "mean_loss.png", format="png")
    plt.show()







