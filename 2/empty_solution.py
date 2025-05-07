import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons
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


def data_generation(samples_number):
    mean = [0, 0]
    cov_mat = [[1, 0.5], [0.5, 1]]
    data = np.random.multivariate_normal(mean, cov_mat, samples_number)
    y = np.ndarray(samples_number)
    for i in range(samples_number):
        y[i] = label_func(data[i])
    y[y == 0] = 1
    return data, y
### Exercise Solution ###

def pratical_1_runner(save_path=None):
    n = len(number_of_samples)
    m = len(regularizations)
    fig, axs = plt.subplots(n, m, figsize=(4 * m, 4 * n), squeeze=False)
    for i, m in enumerate(number_of_samples):
        for j, C in enumerate(regularizations):
            sub_fig = axs[i][j]
            svm_model = SVC(C=C, kernel='linear')
            X, y = data_generation(m)
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

            # Subfig config first (before SVM plotting)
            plt.xlim(x1_min, x1_max)
            plt.ylim(x2_min, x2_max)

            # Plot SVM decision boundary
            x2_range = np.linspace(x2_min, x2_max, 999)
            xx, yy = np.meshgrid(x1_range, x2_range)
            grid = np.c_[xx.ravel(), yy.ravel()]
            Z_svm = svm_model.decision_function(grid)
            Z_svm = Z_svm.reshape(xx.shape)

            # Create proxy artist for SVM boundary
            proxy_svm = mlines.Line2D([], [], color='red', linewidth=2, label='SVM decision boundary')
            sub_fig.contour(xx, yy, Z_svm, levels=[0], colors='red', linewidths=2)

            # Add all elements to legend including proxy_svm
            sub_fig.legend(loc='upper left', handles=[*sub_fig.get_legend_handles_labels()[0], proxy_svm])


    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "practical_1.png"))
        plt.clf()
    else:
        plt.show()


def practical_2_runner(save_path=None):
    pass


if __name__ == "__main__":
    path = None
    pratical_1_runner(save_path=path)
    practical_2_runner(save_path=path)
    pass