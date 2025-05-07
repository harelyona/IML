import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.lines as mlines

number_of_samples = [5, 10, 20, 100]
regularizations = [0.1, 1, 5, 10, 100]
SCATTER_SIZE = 3

def get_svm_hypothesis(svm_model):
    """
    Extracts the SVM hypothesis from a trained SVC model.

    Args:
        svm_model (SVC): A trained SVC model with a linear kernel.

    Returns:
        tuple: A tuple containing the support vectors, dual coefficients, and intercept.
    """
    support_vectors = svm_model.support_vectors_
    dual_coef = svm_model.dual_coef_
    intercept = svm_model.intercept_

    return support_vectors, dual_coef, intercept

### HELPER FUNCTIONS ###
# Add here any helper functions which you think will be useful

def label_func(x):
    return np.sign(x @ np.array([-0.6, 0.4]))


def data_generation1(samples_number, mean, cov_mat):
    data = np.random.multivariate_normal(mean, cov_mat, samples_number)
    y = np.ndarray(samples_number)
    for i in range(samples_number):
        y[i] = label_func(data[i])
    y[y == 0] = 1
    return data, y

def data_generation2(samples_number):
    X_moons, y_moons = make_moons(n_samples=samples_number, noise=0.2)
    X_circles, y_circles = make_circles(n_samples=samples_number, noise=0.1)
    # y[y == 0] = -1
    # return X, y
### Exercise Solution ###

def pratical_1_runner(save_path=None):
    n = len(number_of_samples)
    m = len(regularizations)
    mean = [0, 0]
    cov_mat = [[1, 0.5], [0.5, 1]]
    fig, axs = plt.subplots(n, m, figsize=(4 * m, 4 * n), squeeze=False)
    for i, m in enumerate(number_of_samples):
        for j, C in enumerate(regularizations):
            sub_fig = axs[i][j]
            svm_model = SVC(C=C, kernel='linear')
            X, y = data_generation1(m, mean, cov_mat)
            svm_model.fit(X, y)

            # Plot samples
            sub_fig.scatter(X[y == 1, 0], X[y == 1, 1], label='True samples', s=SCATTER_SIZE)
            sub_fig.scatter(X[y == -1, 0], X[y == -1, 1], label='False samples', s=SCATTER_SIZE)

            # Plot decision boundaries
            padding = 1.0
            x1_min, x1_max = min(X[:, 0]) - padding, max(X[:, 0]) + padding
            x2_min, x2_max = min(X[:, 1]) - padding, max(X[:, 1]) + padding
            x1_range = np.linspace(x1_min, x1_max, 999)
            sub_fig.plot(x1_range, x1_range * 1.5, color='black', linestyle='--', label='decision boundary')

            # Subfig config
            plt.xlim(x1_min, x1_max)
            plt.ylim(x2_min, x2_max)

            w = svm_model.coef_[0]
            b = svm_model.intercept_[0]
            x1_range = np.linspace(x1_min, x1_max, 999)
            x2_range = (-w[0] * x1_range - b) / w[1]
            sub_fig.plot(x1_range, x2_range, color='aquamarine', linewidth=2, label='SVM decision boundary')

            sub_fig.legend(loc='upper left')


    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "practical_1.png"))
        plt.clf()
    else:
        plt.show()


def practical_2_runner(save_path=None):

    if save_path:
        plt.savefig(os.path.join(save_path, "practical_2.png"))
        plt.clf()
    else:
        plt.show()


if __name__ == "__main__":
    path = "plots"
    pratical_1_runner(save_path=path)
    practical_2_runner(save_path=path)
    pass