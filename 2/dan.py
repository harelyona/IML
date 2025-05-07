import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.svm import SVC

# Constants
RAND_SEED = 42
m_lst = [5, 10, 20, 100]
C_lst = [0.1, 1, 5, 10, 100]


def label_func(x):
    inner_prod = x @ np.array([-0.6, 0.4])
    return np.sign(inner_prod)


def generate_dataset(num_samples):
    mean = np.array([0, 0])
    cov = np.array([[1, 0.5], [0.5, 1]])
    dataset = np.random.RandomState(RAND_SEED).multivariate_normal(mean, cov, num_samples)
    labels = label_func(dataset)
    labels = np.array([1 if label == 1 else -1 for label in labels])
    return dataset, labels


def pratical_1_runner(save_path=None):
    num_rows = len(m_lst)
    num_cols = len(C_lst)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), squeeze=False)

    for i, m in enumerate(m_lst):
        for j, C in enumerate(C_lst):
            ax = axs[i][j]
            X, y = generate_dataset(num_samples=m)

            # Train soft-SVM
            soft_svm = SVC(kernel='linear', C=C)
            soft_svm.fit(X, y)

            # Plot scatter plots for true labels
            ax.scatter(X[y == 1, 0], X[y == 1, 1], c='C0', marker='o', edgecolor='k', label='+1 (true)')
            ax.scatter(X[y == -1, 0], X[y == -1, 1], c='C1', marker='s', edgecolor='k', label='-1 (true)')

            # Create a grid for plotting decision boundaries
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                 np.linspace(y_min, y_max, 200))
            grid = np.c_[xx.ravel(), yy.ravel()]

            # Decision boundary from function f
            Z_f = label_func(grid)
            Z_f = Z_f.reshape(xx.shape)
            ax.contour(xx, yy, Z_f, levels=[0], colors='g',
                       linestyles='--', linewidths=2)

            # SVM decision boundary
            Z_svm = soft_svm.decision_function(grid)
            Z_svm = Z_svm.reshape(xx.shape)
            ax.contour(xx, yy, Z_svm, levels=[0], colors='r',
                       linewidths=2)

            # Ensure the point is within the axis limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            # Create proxy artists for the decision boundaries
            proxy_f = mlines.Line2D([], [], color='g', linestyle='--', linewidth=2,
                                      label='f decision boundary')
            proxy_svm = mlines.Line2D([], [], color='r', linewidth=2,
                                        label='SVM decision boundary')

            ax.set_title(f"SVM on m={m}, C={C}")
            ax.set_xlabel('x₁')
            ax.set_ylabel('x₂')
            ax.legend(handles=[proxy_f, proxy_svm], loc='upper left')
            ax.grid(True)

    plt.tight_layout()

    if save_path is not None:
        fname = os.path.join(save_path, "combined_svm_plots.png")
        plt.savefig(fname, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def practical_2_runner(save_path=None):
    pass


if __name__ == "__main__":
    path = '.'
    pratical_1_runner(save_path=None)
    practical_2_runner(save_path=path)