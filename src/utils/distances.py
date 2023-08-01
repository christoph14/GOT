import networkx as nx
import numpy as np
import scipy.linalg as slg

from utils.strategies import get_strategy, get_filters


def compute_distance(graphs, strategy, strategy_args):
    G1, G2 = graphs
    # Compute alignment
    strategy = get_strategy(strategy, it=10, tau=1, n_samples=30, lr=0.2, epochs=1000, **strategy_args)
    L1 = nx.laplacian_matrix(G1).todense()
    L2 = nx.laplacian_matrix(G2).todense()
    m1 = np.array([nx.get_node_attributes(G1, 'labels')[v] for v in G1.nodes])
    m2 = np.array([nx.get_node_attributes(G2, 'labels')[v] for v in G2.nodes])
    P = strategy(L1, L2)
    gL1 = get_filters(L1, strategy_args['filter_name'])
    gL2 = get_filters(L2, strategy_args['filter_name'])
    if (np.isnan(P) | np.isinf(P)).any():
        return {
            'got': np.nan, 'got-label': np.nan,
            'approx': np.nan, 'approx-label': np.nan,
            'fro': np.nan, 'fro-label': np.nan,
        }

    # Create dict for different distance measures
    result = {}
    # Add Wasserstein distance between distributions
    A = gL1 @ gL1
    B = P.T @ gL2 @ P @ P.T @ gL2 @ P
    result['got'] = np.trace(A) + np.trace(B) - 2 * np.trace(slg.sqrtm(A @ B)).real
    result['got-label'] = result['got'] + slg.norm(m1 - P.T @ m2)**2
    # Add approximated Wasserstein distance
    result['approx'] = np.trace(gL1 @ gL1) + np.trace(gL2 @ gL2) - 2 * np.trace(gL1 @ P.T @ gL2 @ P)
    result['approx-label'] = result['approx'] + slg.norm(m1 - P.T @ m2)**2
    # Add Frobenius norm
    result['fro'] = slg.norm(gL1 - P.T @ gL2 @ P, ord='fro')
    result['fro-label'] = result['fro'] + slg.norm(m1 - P.T @ m2)**2
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
    result = np.trace(A) + np.trace(B) - 2 * np.trace(slg.sqrtm(A @ B))
    distance = np.abs(result.real)
    return distance
