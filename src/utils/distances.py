import networkx as nx
import numpy as np
import ot
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
    Root_1 = slg.sqrtm(A)
    result = np.trace(A) + np.trace(B) - 2 * np.trace(slg.sqrtm(Root_1 @ B @ Root_1))
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

    # Define the initial distributions
    p = ot.unif(len(C1))
    q = ot.unif(len(C2))

    distance = ot.gromov_wasserstein2(C1, C2, p, q, loss_fun='square_loss', log=False, armijo=False, G0=None)
    return distance
