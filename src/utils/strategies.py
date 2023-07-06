import networkx as nx
import numpy as np
import ot
import pygmtools as pygm
import scipy.linalg as slg
from scipy.optimize import quadratic_assignment

from alignment import got_strategy, gw_entropic, gw_strategy, ipfp
from alignment._filter_graph_optimal_transport import PstoH, P_nv2, find_trace_sink_wass_filters_reg
from fGOT import fgot_mgd
from fGOT.got_nips import find_permutation
from utils.help_functions import graph_from_laplacian

pygm.BACKEND = 'numpy'


def get_strategy(strategy_name, it, tau, n_samples, epochs, lr, seed=42, verbose=False, alpha=0.1, ones=True,
                 epsilon=0.006, filter_name=None, scale=False):
    """Return a strategy computing a transport plan from L1 to L2."""
    if strategy_name.lower() == 'got':
        def strategy(L1, L2):
            return got_strategy(L1, L2, it, tau, n_samples, epochs, lr, loss_type='w', seed=seed, verbose=verbose,
                                alpha=alpha, ones=ones)
    elif strategy_name.lower() == 'got-original':
        def strategy(L1, L2):
            _, _, permutation = find_permutation(
                L1, L2, it, tau, n_samples, epochs, lr, loss_type='w', alpha=alpha, ones=ones, graphs=True
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
    elif strategy_name.lower().startswith('fgot'):
        if len(strategy_name.split("-")) > 1:
            if filter_name is not None:
                print('The given filter is not used because it is specified by the algo name.')
            filter_name = strategy_name.lower().split("-")[1]
        def strategy(L1, L2):
            # TODO different tau than GOT
            # To avoid "Warning: numerical errors at iteration 0" increase epsilon
            T = find_trace_sink_wass_filters_reg(
                L1, L2, epsilon, filter_name, tau, epochs
            ).T
            if scale:
                T *= np.sqrt(len(L1) * len(L2))
            return T
    elif strategy_name.lower() == 'gw':
        def strategy(L1, L2):
            T = gw_strategy(L1, L2)
            if scale:
                T *= np.sqrt(len(L1) * len(L2))
            return T
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
        def strategy(L1, L2):
            return ipfp(L1, L2).T
    elif strategy_name.lower() == 'ipfp-got':
        def strategy(L1, L2):
            return ipfp(L1, L2, affinity='got').T
    elif strategy_name.lower() == 'pstoh':
        def strategy(L1, L2):
            return PstoH(L1, L2, it=it, tau=tau) * len(L1)
    elif strategy_name.lower() == 'p_nv2':
        def strategy(L1, L2):
            return P_nv2(L1, L2, it=it, tau=tau) * len(L1)
    # elif strategy_name.lower() == 'qap':
    #     # TODO remove (include in next?)
    #     def strategy(L1, L2):
    #         L1_inv = slg.pinv(L1)
    #         L2_inv = slg.pinv(L2)
    #         res = quadratic_assignment(L2_inv.T, L1_inv, method='2opt', options={'maximize': True})
    #         P = np.eye(len(L1), dtype=int)[res['col_ind']]
    #         return P
    elif strategy_name.lower().startswith('qap'):
        if len(strategy_name.split("-")) > 1:
            if filter_name is not None:
                print('The given filter is not used because it is specified by the algo name.')
            filter_name = strategy_name.split("-")[1]
        def strategy(L1, L2):
            # Add dummy nodes
            n = max(len(L1), len(L2))
            L1_ = np.zeros((n,n))
            L2_ = np.zeros((n,n))
            L1_[:len(L1), :len(L1)] = L1
            L2_[:len(L2), :len(L2)] = L2
            g1, g2 = get_filters(L1_, filter_name), get_filters(L2_, filter_name)
            res = quadratic_assignment(g2.T, g1, method='2opt', options={'maximize': True})
            P = np.eye(n, dtype=int)[res['col_ind']]
            P = P[:len(L2), :len(L1)]
            return P
    elif strategy_name.lower().startswith('slsqp'):
        if len(strategy_name.split("-")) > 1:
            if filter_name is not None:
                print('The given filter is not used because it is specified by the algo name.')
            filter_name = strategy_name.split("-")[1]
        def strategy(L1, L2):
            from scipy.optimize import minimize

            n1 = len(L1)
            n2 = len(L2)
            g1, g2 = get_filters(L1, filter_name), get_filters(L2, filter_name)
            K = np.kron(g2, g1)

            def objective_function(x):
                return - (x.reshape(1, -1) @ K @ x.reshape(-1, 1)).item()

            x0 = np.eye(n2, n1).flatten()

            bounds = [(0, None) for _ in range(x0.size)]

            constraints = [
                {'type': 'eq', 'fun': lambda x: x.reshape(n2, n1) @ np.ones(n1) - np.sqrt(n1 * n2) / n2},
                {'type': 'eq', 'fun': lambda x: x.reshape(n2, n1).T @ np.ones(n2) - np.sqrt(n1 * n2) / n1}
            ]

            res = minimize(objective_function, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            P = res['x'].reshape(n2, n1)
            return P
    elif strategy_name.lower().startswith('blowup-qap'):
        if len(strategy_name.split("-")) > 2:
            if filter_name is not None:
                print('The given filter is not used because it is specified by the algo name.')
            filter_name = strategy_name.split("-")[2]
        def strategy(L1, L2):
            import math
            n = math.lcm(len(L1), len(L2))
            L1_ = np.zeros((n,n))
            L2_ = np.zeros((n,n))
            for i in range(round(n / len(L1))):
                L1_[i * len(L1): (i+1) * len(L1), i * len(L1): (i+1) * len(L1)] = L1
            for i in range(round(n / len(L2))):
                L2_[i * len(L2): (i+1) * len(L2), i * len(L2): (i+1) * len(L2)] = L2
            g1, g2 = get_filters(L1_, filter_name), get_filters(L2_, filter_name)
            res = quadratic_assignment(g2.T, g1, method='2opt', options={'maximize': True})
            P_blowup = np.eye(n, dtype=int)[res['col_ind']]
            P = np.zeros((len(L2), len(L1)))
            for i in range(n):
                for j in range(n):
                    P[i % len(L2), j % len(L1)] += P_blowup[i,j]
            P = P / n * np.sqrt(len(L1) * len(L2))
            return P
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
