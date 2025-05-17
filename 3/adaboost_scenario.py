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
DECISION_SURFACE_LABEL = "Decision Surface"
DECISION_SURFACE_COLOR = "black"
TRUE_LABEL_COLOR = "green"
FALSE_LABEL_COLOR = "red"
SCATTER_SIZE = 3
FIG_SIZE = (10, 10)


def plot_decision_surface(plot, model: AdaBoost, t, xrange, yrange, density=120, dotted=False):
    xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = model.partial_predict(np.c_[xx.ravel(), yy.ravel()], t)

    plot.contour(xx, yy, pred.reshape(xx.shape), levels=[0], colors=DECISION_SURFACE_COLOR)
    plot.plot([], [], color=DECISION_SURFACE_COLOR, label=DECISION_SURFACE_LABEL)

    if dotted:
        plot.scatter(x=xx.ravel(), y=yy.ravel(), opacity=1, mode="markers",
                     hoverinfo="skip", showlegend=False, label=DECISION_SURFACE_LABEL)

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
    plt.savefig(r"plots/question1.png")
    plt.clf()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    all_samples = np.append(train_X, test_X, axis=0)
    all_labels = np.append(train_y, test_y, axis=0)
    cmap = np.where(all_labels == 1, TRUE_LABEL_COLOR, FALSE_LABEL_COLOR)
    fig, axs = plt.subplots(2, 2, figsize=FIG_SIZE)
    for i, t in enumerate(T):
        ax = axs[i // 2][i % 2]
        ax.scatter(all_samples[:, 0], all_samples[:, 1], c=cmap, s=SCATTER_SIZE, label="Samples")
        plot_decision_surface(ax, model, t, lims[0], lims[1])
        loss = model.partial_loss(all_samples, all_labels, t)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(f"Decision Surface with {t} Learners\n loss: {loss:.3f}")
        ax.legend()
    plt.tight_layout()
    fig.savefig(r"plots/question2.png")
    fig.clf()

    # Question 3: Decision surface of best performing ensemble
    raise NotImplementedError()

    # Question 4: Decision surface with weighted samples
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0,)
    # x_range = (-3, 3)
    # y_range = (-3, 3)
    # predict = lambda X: np.where(np.sum(X**2, axis=1) > 1, 1, -1)
    # x_data = np.linspace(-10, 10, 9999)
    # y_data = np.linspace(-10, 10, 9999)
    # xx, yy = np.meshgrid(x_data, y_data)
    # pred = predict(np.c_[xx.ravel(), yy.ravel()])
    # plt.scatter(x_data, y_data, label="Data")
    # plot_decision_surface(predict, x_range, y_range)
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.title("Decision Surface")
    # plt.legend()
    # plt.show()