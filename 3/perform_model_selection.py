import matplotlib.pyplot as plt
from sklearn import datasets
from cross_validate import cross_validate
from estimators import RidgeRegression, Lasso, LinearRegression
from adaboost_scenario import plot_config, FIG_SIZE
from utils import *

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
    X = pd.DataFrame(data.data)
    y = pd.Series(data.target)
    X_train, y_train, X_test, y_test = split_train_test(X, y, train_proportion=n_samples / X.shape[0])

    # Question 2 - Perform Cross Validation for different values of the regularization parameter for Ridge and
    # Lasso regressions
    lasso_train_errors = np.zeros(n_evaluations)
    lasso_validation_errors = np.zeros(n_evaluations)
    ridge_train_errors = np.zeros(n_evaluations)
    ridge_validation_errors = np.zeros(n_evaluations)
    ridge_regulazations = np.insert(np.linspace(0, 0.07, num=n_evaluations - 1, dtype=float), 0, 0)
    lasso_regulazations = np.insert(np.linspace(0, 0.4, num=n_evaluations - 1, dtype=float), 0, 0)
    fig, (lasso_plot, ridge_plot) = plt.subplots(2, 1, figsize=FIG_SIZE)
    for i in range(n_evaluations):
        lasso = Lasso(float(lasso_regulazations[i]))
        ridge = RidgeRegression(float(ridge_regulazations[i]))
        lasso_train_errors[i], lasso_validation_errors[i] = cross_validate(lasso, X_train, y_train)
        ridge_train_errors[i], ridge_validation_errors[i] = cross_validate(ridge, X_train, y_train)
    lasso_plot.scatter(lasso_regulazations, lasso_train_errors, label="Lasso Train Error")
    lasso_plot.scatter(lasso_regulazations, lasso_validation_errors, label="Lasso Validation Error")
    ridge_plot.scatter(ridge_regulazations, ridge_train_errors, label="Ridge Train Error")
    ridge_plot.scatter(ridge_regulazations, ridge_validation_errors, label="Ridge Validation Error")
    plot_config(lasso_plot, f"Lassos Error Vs Regulazations\n    number of evaluations = {n_evaluations}", "Regularization Parameter", "Error", False)
    plot_config(ridge_plot, f"Ridges Error Vs Regulazations\n    number of evaluations = {n_evaluations}", "Regularization Parameter", "Error", False)
    plt.tight_layout()
    fig.show()
    plt.clf()
    best_lasso_validation_error = np.min(lasso_validation_errors)
    best_ridge_validation_error = np.min(ridge_validation_errors)
    best_lasso_regularization = lasso_regulazations[np.argmin(lasso_validation_errors)]
    best_ridge_regularization = ridge_regulazations[np.argmin(ridge_validation_errors)]
    print(f"Best Lasso Validation Error: {best_lasso_validation_error} With Regularization: {best_lasso_regularization}")
    print(f"Best Ridge Validation Error: {best_ridge_validation_error} With Regularization: {best_ridge_regularization}")

    # Fitted Using 10000 evaluations
    # Best Lasso Validation Error: 3704.3809241002186 With Regularization: 0.30213879058316667
    # Best Ridge Validation Error: 3913.9904684198737 With Regularization: 0.04255577781197108
    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    lasso_model = Lasso(best_lasso_regularization)
    ridge_model = RidgeRegression(best_ridge_regularization)
    least_squares_model = LinearRegression()

    lasso_model.fit(X_train, y_train)
    ridge_model.fit(X_train, y_train)
    least_squares_model.fit(X_train, y_train)

    lasso_test_error = lasso_model.loss(X_test, y_test)
    ridge_test_error = ridge_model.loss(X_test, y_test)
    least_squares_test_error = least_squares_model.loss(X_test, y_test)

    print(f"lasso test error: {lasso_test_error}")
    print(f"ridge test error: {ridge_test_error}")
    print(f"least squares test error: {least_squares_test_error}")






if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
