import networkx as nx
import numpy as np
import scipy.linalg as slg


def wasserstein_distance(A, B):
    """Calculate the wasserstein distance between Gaussian distributions.

    Parameters
    ----------
    A : array-like of shape (n, n)
        Covariance matrix of the first distribution.
    B : array-like of shape (n, n)
        Covariance matrix of the first distribution.

    Returns
    -------
    distance : float
        Calculated Wasserstein distance.
    """
    root = slg.sqrtm(A)
    result = np.trace(A) + np.trace(B) - 2 * np.trace(slg.sqrtm(root @ B @ root))
    distance = np.abs(result.real)
    return distance


def gw_distance(G1, G2):
    """Compute the Gromov-Wasserstein distance between two graphs.
    Use the shortest paths as structure similarity matrices.

    Parameters
    ----------
    G1 : nx.graph
        First graph.
    G2 : nx.graph
        Second graph.

    Returns
    -------
    distance : float
        The computed Gromov-Wasserstein distance between G1 and G2.
    """

    # Compute the shortest path matrices
    C1 = nx.floyd_warshall_numpy(G1)
    C2 = nx.floyd_warshall_numpy(G2)

    n = len(C1)
    distance = (np.linalg.norm(C1 - C2, ord='fro') / n)**2
    return distance
