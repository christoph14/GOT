import networkx as nx
import numpy as np


def remove_edges(L, block_size, between_probability, within_probability=0.5, seed=None):
    n = L.shape[0]
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = seed
    A = -L.copy()
    np.fill_diagonal(A, 0)

    for i in range(n):
        for j in range(i, n):
            if np.floor(i / block_size) == np.floor(j / block_size):
                removal_probability = within_probability
            else:
                removal_probability = between_probability
            if rng.random() < removal_probability:
                A[i, j] = 0
                A[j, i] = 0
    L_reduced = nx.laplacian_matrix(nx.from_numpy_array(A))
    L_reduced = np.double(np.array(L_reduced.todense()))
    return L_reduced


def regularise_and_invert(x, y, alpha, ones):
    x_reg = regularise_invert_one(x, alpha, ones)
    y_reg = regularise_invert_one(y, alpha, ones)
    return [x_reg, y_reg]


def regularise_invert_one(x, alpha, ones):
    if ones:
        x_reg = np.linalg.inv(x + alpha * np.eye(len(x)) + np.ones([len(x), len(x)])/len(x))
    else:
        x_reg = np.linalg.pinv(x) + alpha * np.eye(len(x))
    return x_reg
