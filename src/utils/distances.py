import networkx as nx
import numpy as np
import scipy.linalg as slg

from utils.strategies import get_strategy, get_filters


def compute_distance(G1, G2, strategy, strategy_args, distance=None):
    strategy = get_strategy(strategy, it=10, tau=1, n_samples=30, lr=0.2, epochs=1000, **strategy_args)
    L1 = nx.laplacian_matrix(G1).todense()
    L2 = nx.laplacian_matrix(G2).todense()
    P = strategy(L1, L2)
    gL1 = get_filters(L1, strategy_args['filter_name'])
    gL2 = get_filters(L2, strategy_args['filter_name'])
    if distance == 'got':
        result = np.trace(gL1**2) + np.trace(gL2**2) - 2 * np.trace(np.sqrt(gL1 @ P.T @ gL2 @ P))
    elif distance == 'fro':
        result = np.trace(gL1**2) + np.trace(gL2**2) - 2 * np.trace(gL1 @ P.T @ gL2 @ P)
    return result


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
