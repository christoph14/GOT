import warnings

import networkx as nx
import numpy as np


def remove_edges(G, communities, between_probability, within_probability=0.5, seed=None):
    if not nx.is_connected(G):
        raise ValueError("G must be connected.")

    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, int):
        rng = np.random.default_rng(seed)
    else:
        rng = seed

    G_new = G.copy()

    # Count number of edges within and between communities
    n_within = 0
    n_between = 0
    for u, v in G_new.edges:
        if communities[u] == communities[v]:
            n_within += 1
        else:
            n_between += 1

    # Remove edges
    removed_within = 0
    removed_between = 0
    for u, v in rng.permutation(G_new.edges):
        if communities[u] == communities[v]:
            if removed_within < np.floor(n_within * within_probability):
                G_new.remove_edge(u, v)
                if nx.is_connected(G_new):
                    removed_within += 1
                else:
                    G_new.add_edge(u, v)
        else:
            if removed_between < np.floor(n_between * between_probability):
                G_new.remove_edge(u, v)
                if nx.is_connected(G_new):
                    removed_between += 1
                else:
                    G_new.add_edge(u, v)

    # Check if enough edges could be removed
    if removed_within < np.floor(n_within * within_probability):
        warnings.warn("Could not remove enough edges within the communities.")
    if between_probability < np.floor(n_between * between_probability):
        warnings.warn("Could not remove enough edges between the communities.")
    return G_new


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


def graph_from_laplacian(L):
    A = -L.copy()
    np.fill_diagonal(A, 0)
    G = nx.from_numpy_array(A)
    return G


def random_permutation(n, random_seed=None):
    if random_seed is None:
        rng = np.random.default_rng()
    elif isinstance(random_seed, int):
        rng = np.random.default_rng(random_seed)
    elif isinstance(random_seed, np.random.Generator):
        rng = random_seed
    else:
        raise ValueError("random_seed must be None, int, or np.random.Generator.")
    idx = rng.permutation(n)
    permutation = np.eye(n)
    permutation = permutation[idx, :]
    return permutation
