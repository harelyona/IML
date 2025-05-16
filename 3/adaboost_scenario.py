import numpy as np
from typing import Tuple

from matplotlib import pyplot as plt

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adaboost import AdaBoost
from decision_stump import DecisionStump
import plotly.io as pio
pio.renderers.default = "browser"

def plot_decision_surface(predict, xrange, yrange, density=120, dotted=False, colorscale=custom, showscale=True):
    xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = predict(np.c_[xx.ravel(), yy.ravel()])

    if dotted:
        plt.scatter(x=xx.ravel(), y=yy.ravel(), opacity=1, mode="markers", hoverinfo="skip", showlegend=False)
    plt.contour(xrange, yrange, pred.reshape(xx.shape))

def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    training_errors = np.zeros(n_learners)
    test_errors = np.zeros(n_learners)
    for size in range(1, n_learners + 1):
        training_errors[size - 1] = model.partial_loss(train_X, train_y, size)
        test_errors[size - 1] = model.partial_loss(test_X, test_y, size)
    plt.plot(range(1, n_learners + 1), training_errors, label="Training Error")
    plt.plot(range(1, n_learners + 1), test_errors, label="Test Error")
    plt.xlabel("Number of Learners")
    plt.ylabel("Error")
    plt.title("Training and Test Errors of AdaBoost")
    plt.legend()
    plt.savefig(r"plots/adaboost errors.png")
    plt.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    raise NotImplementedError()

    # Question 3: Decision surface of best performing ensemble
    raise NotImplementedError()

    # Question 4: Decision surface with weighted samples
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0,)
    x_range = (-3, 3)
    y_range = (-3, 3)
    predict = lambda x: np.sign(np.max(x, axis=1))  # Dummy prediction function
    plot_decision_surface(predict, x_range, y_range, True)
    plt.show()
    # X = np.array([[1, 2], [2, 3], [3, 4]])
    # y = np.array([1, -1, 1])
    # model = AdaBoost(DecisionStump, 3)
    # model.fit(X, y)
    # surface = decision_surface(
    #     predict=model.predict,  # prediction function of your classifier
    #     xrange=(-1, 5),  # x-axis range
    #     yrange=(-1, 5),  # y-axis range
    #     density=120,  # number of points to evaluate (higher = smoother but slower)
    #     dotted=False,  # True for scatter plot, False for contour plot
    #     colorscale=custom,  # custom color scale defined in utils.py
    #     showscale=True  # whether to show the color scale bar
    # )
    # fig = go.Figure(surface)
    # fig.show()