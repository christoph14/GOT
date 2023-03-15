import networkx as nx
import numpy as np
import ot
import pygmtools as pygm
import scipy.linalg as slg
import torch

from alignment import got_strategy, gw_strategy
from fGOT import fgot_mgd
from fGOT.got_nips import find_permutation
from utils.help_functions import graph_from_laplacian

torch.set_default_tensor_type('torch.DoubleTensor')
pygm.BACKEND = 'numpy'


def get_strategy(strategy_name, it, tau, n_samples, epochs, lr, seed=42, verbose=False, alpha=0.0, ones=True):
    """Return a strategy computing a transport plan from L1 to L2."""
    if strategy_name.lower() == 'got':
        def strategy(L1, L2):
            return got_strategy(L1, L2, it, tau, n_samples, epochs, lr, loss_type='w', seed=seed, verbose=verbose,
                                alpha=alpha, ones=ones)
    elif strategy_name.lower() == 'got-original-l':
        def strategy(L1, L2):
            _, _, permutation = find_permutation(
                L1, L2, it, tau, n_samples, epochs, lr, loss_type='w', alpha=alpha, ones=ones, graphs=False
            )
            return permutation
    elif strategy_name.lower() == 'l2':
        def strategy(L1, L2):
            return got_strategy(L1, L2, it, tau, n_samples, epochs, lr, loss_type='l2', seed=seed, verbose=verbose,
                                alpha=alpha, ones=ones)
    elif strategy_name.lower() == 'l2-inv':
        def strategy(L1, L2):
            return got_strategy(L1, L2, it, tau, n_samples, epochs, lr, loss_type='l2-inv', seed=seed, verbose=verbose,
                                alpha=alpha, ones=ones)
    elif strategy_name.lower() == 'fgot':
        def strategy(L1, L2, epsilon=0.006, method='got'):
            # To avoid "Warning: numerical errors at iteration 0" increase epsilon
            max_iter = 1000
            tol = 1e-9
            n = len(L1)
            m = len(L2)
            p = np.repeat(1 / n, n)
            q = np.repeat(1 / m, m)
            g1 = get_filters(L1, method, tau)
            g2 = get_filters(L2, method, tau)

            gw, log = fgot_mgd.fgot(g1, g2, p, q, epsilon * np.max(g1) * np.max(g2) / n, max_iter=max_iter, tol=tol,
                                    verbose=False, log=True, lapl=True)
            gw *= n
            return gw.T
    elif strategy_name.lower() == 'gw':
        def strategy(L1, L2):
            return gw_strategy(L1, L2)
    elif strategy_name.lower() == 'rrmw':
        # Reweighted Random Walks for Graph Matching
        def strategy(L1, L2):
            G1 = graph_from_laplacian(L1)
            G2 = graph_from_laplacian(L2)
            n1 = G1.number_of_nodes()
            n2 = G2.number_of_nodes()
            A1 = nx.adjacency_matrix(G1).todense()
            A2 = nx.adjacency_matrix(G2).todense()

            conn1, edge1 = pygm.utils.dense_to_sparse(A1)
            conn2, edge2 = pygm.utils.dense_to_sparse(A2)
            import functools
            gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1)  # set affinity function
            K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, [n1], None, [n2], None,
                                         edge_aff_fn=gaussian_aff)
            X = pygm.rrwm(K, n1, n2) * n1
            X = pygm.hungarian(X)
            return X.T
    elif strategy_name.lower() == 'ipfp':
        # Integer Projected Fixed Point from
        # "An integer projected fixed point method for graph matching and map inference."
        def strategy(L1, L2):
            G1 = graph_from_laplacian(L1)
            G2 = graph_from_laplacian(L2)
            n1 = G1.number_of_nodes()
            n2 = G2.number_of_nodes()
            A1 = nx.adjacency_matrix(G1).todense()
            A2 = nx.adjacency_matrix(G2).todense()

            conn1, edge1 = pygm.utils.dense_to_sparse(A1)
            conn2, edge2 = pygm.utils.dense_to_sparse(A2)
            import functools
            gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1)  # set affinity function
            K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, [n1], None, [n2], None,
                                         edge_aff_fn=gaussian_aff)
            X = pygm.ipfp(K, n1, n2) * n1
            X = pygm.hungarian(X)
            return X.T
    elif strategy_name.lower() == 'random':
        def strategy(L1, L2):
            rng = np.random.default_rng(seed)
            n = L1.shape[0]
            idx = rng.permutation(n)
            P = np.eye(n)
            P = P[idx, :]
            return P
    else:
        raise NotImplementedError(
            "Only strategies 'GOT', 'L2', 'L2-inv', fGOT, GW, RRMW, IPFP, and random are implemented."
        )
    return strategy


def gw_strategy_entropic(L1, L2, epsilon=0.04, max_iter=2000):
    """Determines the permutation matrix for two given graphs.

    Parameters
    ----------
    L1 : array-like of shape (n, n)
        Laplacian matrix of the first graph.
    L2 : array-like of shape (n, n)
        Laplacian matrix of the second graph.
    epsilon : float
        Regularization term > 0.
    max_iter : int, default=2000
        Max number of iterations.
    
    Returns
    -------
    transportation_matrix : numpy.ndarray of shape (n, n)
        The calculated transportation matrix.
    """
    G1 = nx.from_numpy_array(np.diag(np.diag(L1)) - L1)
    C1 = nx.floyd_warshall_numpy(G1)
    G2 = nx.from_numpy_array(np.diag(np.diag(L2)) - L2)
    C2 = nx.floyd_warshall_numpy(G2)

    p = ot.unif(len(C1))
    q = ot.unif(len(C2))
    gw, log = ot.gromov.entropic_gromov_wasserstein(
        C1, C2, p, q, 'square_loss', epsilon=epsilon, max_iter=max_iter, log=True, verbose=False
    )
    return len(L1) * gw


def get_filters(L1, method, tau=0.2):
    if method == 'got':
        g1 = np.real(slg.sqrtm(fgot_mgd.regularise_invert_one(L1, alpha=0.1, ones=False)))
    elif method == 'weight':
        g1 = np.diag(np.diag(L1)) - L1
    elif method == 'heat':
        g1 = slg.expm(-tau * L1)
    elif method == 'sqrtL':
        g1 = np.real(slg.sqrtm(L1))
    elif method == 'L':
        g1 = L1
    elif method == 'sq':
        g1 = L1 @ L1
    else:
        raise ValueError("The given method is not valid.")
    return g1
