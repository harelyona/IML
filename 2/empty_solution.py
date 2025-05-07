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
TRUE_LABEL_COLOR = "#2ECC71"
FALSE_LABEL_COLOR = "#E74C3C"
DECISION_MODEL_COLOR = "#3498DB"
REAL_DECISION_COLOR = "#2C3E50"
TRUE_COLOR_BACKGROUND = "#ABEBC6"
FALSE_COLOR_BACKGROUND = "#FADBD8"
FIGURE_SIZE_PART_2 = (12, 12)
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


def generate_2D_gaussian_distribution(samples_number, mean, cov_mat):
    data = np.random.multivariate_normal(mean, cov_mat, samples_number)
    y = np.ndarray(samples_number)
    for i in range(samples_number):
        y[i] = label_func(data[i])
    y[y == 0] = 1
    return data, y

def data_generation2(samples_of_each):
    cov_mat = np.array([[0.5, 0.2], [0.2, 0.5]])
    X_moons, y_moons = make_moons(n_samples=samples_of_each, noise=0.2)
    X_circles, y_circles = make_circles(n_samples=samples_of_each, noise=0.1)
    X_gauss1, y_gauss1 = generate_2D_gaussian_distribution(samples_of_each, [-1, -1], cov_mat)
    X_gauss2, y_gauss2 = generate_2D_gaussian_distribution(samples_of_each, [1, 1], cov_mat)
    X = np.concatenate((X_gauss1, X_gauss2, X_moons, X_circles), axis=0)
    y = np.concatenate((y_gauss1, y_gauss2, y_moons, y_circles), axis=0)
    return X, y


def plot_model(ax, model, X, y):
    # Plot data points
    ax.scatter(X[y == 1, 0], X[y == 1, 1], label='True samples', s=SCATTER_SIZE, color=TRUE_LABEL_COLOR)
    ax.scatter(X[y == -1, 0], X[y == -1, 1], label='False samples', s=SCATTER_SIZE, color=FALSE_LABEL_COLOR)

    # Create mesh grid for background coloring
    padding = 1.0
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000),
                         np.linspace(y_min, y_max, 1000))

    # Get model predictions for mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Color background according to predictions
    ax.contourf(xx, yy, Z, alpha=0.2,
                colors=[FALSE_COLOR_BACKGROUND, TRUE_COLOR_BACKGROUND])

    # Add decision boundary
    boundary = ax.contour(xx, yy, Z, colors=DECISION_MODEL_COLOR)

    # Create custom legend handles
    legend_elements = [
        plt.scatter([], [], c=TRUE_LABEL_COLOR, s=SCATTER_SIZE, label='True samples'),
        plt.scatter([], [], c=FALSE_LABEL_COLOR, s=SCATTER_SIZE, label='False samples'),
        mlines.Line2D([], [], color=DECISION_MODEL_COLOR, label='Decision boundary')
    ]

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.legend(handles=legend_elements, loc='upper left')
    ax.grid(True)/


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
            X, y = generate_2D_gaussian_distribution(m, mean, cov_mat)
            svm_model.fit(X, y)

            # Plot samples
            sub_fig.scatter(X[y == 1, 0], X[y == 1, 1], label='True samples', s=SCATTER_SIZE, color=TRUE_LABEL_COLOR)
            sub_fig.scatter(X[y == -1, 0], X[y == -1, 1], label='False samples', s=SCATTER_SIZE, color=FALSE_LABEL_COLOR)

            # Plot decision boundaries
            padding = 1.0
            x1_min, x1_max = min(X[:, 0]) - padding, max(X[:, 0]) + padding
            x2_min, x2_max = min(X[:, 1]) - padding, max(X[:, 1]) + padding
            x1_range = np.linspace(x1_min, x1_max, 999)
            sub_fig.plot(x1_range, x1_range * 1.5, color=REAL_DECISION_COLOR, linestyle='--', label='decision boundary')

            # Subfig config
            plt.xlim(x1_min, x1_max)
            plt.ylim(x2_min, x2_max)

            w = svm_model.coef_[0]
            b = svm_model.intercept_[0]
            x1_range = np.linspace(x1_min, x1_max, 999)
            x2_range = (-w[0] * x1_range - b) / w[1]
            sub_fig.plot(x1_range, x2_range, color= DECISION_MODEL_COLOR, linewidth=2, label='SVM decision boundary')
            sub_fig.set_title(f"SVM on m={m}, C={C}")
            sub_fig.legend(loc='upper left')
            sub_fig.set_xlabel('x₁')
            sub_fig.set_ylabel('x₂')
            sub_fig.grid(True)


    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "practical_1.png"))
        plt.clf()
    else:
        plt.show()


def practical_2_runner(save_path=None):
    m = 3
    n = 1
    X, y = data_generation2(200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    svm_model = SVC(kernel='linear', C=0.2)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    dt_model = DecisionTreeClassifier(max_depth=7)
    svm_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)
    fig, (ax1, ax2, ax3) = plt.subplots(m, n,figsize=FIGURE_SIZE_PART_2,)
    plot_model(ax1, svm_model, X_train, y_train)
    plot_model(ax2, knn_model, X_train, y_train)
    plot_model(ax3, dt_model, X_train, y_train)
    
    if save_path:
        plt.savefig(os.path.join(save_path, "practical_2.png"))
        plt.clf()
    else:
        plt.show()


if __name__ == "__main__":
    path = "plots"
    #pratical_1_runner(save_path=path)
    practical_2_runner(save_path=None)
    pass