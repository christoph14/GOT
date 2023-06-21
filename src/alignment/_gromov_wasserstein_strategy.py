import networkx as nx
import ot

from utils.help_functions import graph_from_laplacian


def gw_strategy(L1, L2):
    """Determines the permutation matrix for two given graphs.

    Parameters
    ----------
    L1 : array-like of shape (n, n)
        Laplacian matrix of the first graph.
    L2 : array-like of shape (n, n)
        Laplacian matrix of the second graph.

    Returns
    -------
    T : numpy.ndarray of shape (n, n)
        The calculated permutation matrix.
    """
    G1 = graph_from_laplacian(L1)
    C1 = nx.floyd_warshall_numpy(G1)
    G2 = graph_from_laplacian(L2)
    C2 = nx.floyd_warshall_numpy(G2)

    p = ot.unif(len(C1))
    q = ot.unif(len(C2))

    T = ot.gromov_wasserstein(C1, C2, p, q, log=False)
    T = T.T

    return T


def gw_entropic(L1, L2, epsilon=2e-2):
    """Determines the permutation matrix for two given graphs.

    Parameters
    ----------
    L1 : array-like of shape (n, n)
        Laplacian matrix of the first graph.
    L2 : array-like of shape (n, n)
        Laplacian matrix of the second graph.
    epsilon : float
        Regularization term >0

    Returns
    -------
    T : numpy.ndarray of shape (n, n)
        The calculated permutation matrix.
    """
    G1 = graph_from_laplacian(L1)
    C1 = nx.floyd_warshall_numpy(G1)
    G2 = graph_from_laplacian(L2)
    C2 = nx.floyd_warshall_numpy(G2)

    p = ot.unif(len(C1))
    q = ot.unif(len(C2))

    T = ot.gromov.entropic_gromov_wasserstein(
        C1, C2, p, q, 'square_loss', epsilon=epsilon, log=False, verbose=False
    )
    T = T.T

    return T
