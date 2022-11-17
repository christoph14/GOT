import networkx as nx
import numpy as np


def remove_edges(G, communities, between_probability, within_probability=0.5, seed=None):
    if not nx.is_connected(G):
        raise ValueError("G must be connected.")

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = seed

    G_new = G.copy()
    for u, v in G_new.edges:
        if communities[u] == communities[v]:
            p = within_probability
        else:
            p = between_probability

        # Remove edge with given probability
        if rng.random() < p:
            G_new.remove_edge(u, v)

        if not nx.is_connected(G_new):
            G_new.add_edge(u, v)
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
