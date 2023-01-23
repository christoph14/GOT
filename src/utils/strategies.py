import networkx as nx
import numpy as np
import ot
import pygmtools as pygm
import scipy.linalg as slg
import torch

from fGOT import fgot_mgd
from utils.gromov_wasserstein_strategy import gw_strategy
from utils.help_functions import regularise_and_invert, graph_from_laplacian

torch.set_default_tensor_type('torch.DoubleTensor')
pygm.BACKEND = 'numpy'


def get_strategy(strategy_name, it, tau, n_samples, epochs, lr, seed=42, verbose=False, alpha=0.0, ones=True):
    if strategy_name == 'GOT':
        def strategy(L1, L2):
            return got_strategy(L1, L2, it, tau, n_samples, epochs, lr, loss_type='w', seed=seed, verbose=verbose,
                                alpha=alpha, ones=ones)
    elif strategy_name == 'L2':
        def strategy(L1, L2):
            return got_strategy(L1, L2, it, tau, n_samples, epochs, lr, loss_type='l2', seed=seed, verbose=verbose,
                                alpha=alpha, ones=ones)
    elif strategy_name == 'L2-inv':
        def strategy(L1, L2):
            return got_strategy(L1, L2, it, tau, n_samples, epochs, lr, loss_type='l2-inv', seed=seed, verbose=verbose,
                                alpha=alpha, ones=ones)
    elif strategy_name == 'fGOT':
        def strategy(L1, L2, epsilon=0.006, method='got'):
            # To avoid "Warning: numerical errors at iteration 0" increase epsilon
            n = len(L1)
            m = len(L2)
            p = np.repeat(1 / n, n)
            q = np.repeat(1 / m, m)
            max_iter = 500
            g1 = get_filters(L1, method, tau)
            g2 = get_filters(L2, method, tau)

            gw, log = fgot_mgd.fgot(g1, g2, p, q, epsilon * np.max(g1) * np.max(g2) / n, max_iter=max_iter, tol=1e-9,
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
    elif strategy_name == 'random':
        def strategy(L1, L2):
            rng = np.random.default_rng(seed)
            n = L1.shape[0]
            idx = rng.permutation(n)
            P = np.eye(n)
            P = P[idx, :]
            return P
    else:
        raise NotImplementedError(
            "Only strategies 'GOT', 'L2', 'L2-inv', fGOT, GW, RRMW, IPFP, random are implemented."
        )
    return strategy


def got_strategy(L1, L2, it, tau, n_samples, epochs, lr, loss_type='w', seed=42, verbose=True, alpha=0.0, ones=True):
    L1_inv, L2_inv = regularise_and_invert(L1, L2, alpha, ones)

    # Initialization
    torch.manual_seed(seed)
    n = L1.shape[0]
    mean = torch.rand(n, n, requires_grad=True)
    std = 10 * torch.ones(n, n)
    std = std.requires_grad_()
    params = []
    if loss_type == 'w':
        params = wasserstein_initialisation(L1_inv, L2_inv)

    # Optimization
    optimizer = torch.optim.Adam([mean, std], lr=lr, amsgrad=True)
    history = []
    for epoch in range(epochs):
        cost = 0
        cost_vec = np.zeros((1, n_samples))
        for sample in range(n_samples):
            # Sampling
            eps = torch.randn(n, n)
            P_noisy = mean + std * eps  # torch.log(1+torch.exp(std)) * eps

            # Cost function
            DS = doubly_stochastic(P_noisy, tau, it)
            cost = cost + loss(DS, L1, L2, L1_inv, L2_inv, params, loss_type)
            cost_vec[0, sample] = loss(DS, L1, L2, L1_inv, L2_inv, params, loss_type)
        cost = cost / n_samples

        # Gradient step
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Tracking
        history.append(cost.item())
        if verbose and (epoch == 0 or (epoch + 1) % 100 == 0):
            print('[Epoch %4d/%d] loss: %f - std: %f' % (epoch + 1, epochs, cost.item(), std.detach().mean()))

    # PyTorch -> NumPy
    P = doubly_stochastic(mean, tau, it)
    P = P.squeeze()
    P = P.detach().numpy()

    # Keep the max along the rows
    idx = P.argmax(1)
    P = np.zeros_like(P)
    P[range(n), idx] = 1.

    # Convergence plot
    # if plot:
    #     plt.plot(history)
    #     plt.show()

    return P


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


def doubly_stochastic(P, tau, it):
    """Uses logsumexp for numerical stability."""

    A = P / tau
    for i in range(it):
        A = A - A.logsumexp(dim=1, keepdim=True)
        A = A - A.logsumexp(dim=0, keepdim=True)
    return torch.exp(A)


def wasserstein_initialisation(L1_inv, L2_inv):
    # Wasserstein directly on covariance
    Root_1 = slg.sqrtm(L1_inv)
    Root_2 = slg.sqrtm(L2_inv)
    C1_tilde = torch.from_numpy(Root_1.astype(np.double))
    C2_tilde = torch.from_numpy(Root_2.astype(np.double))
    return [C1_tilde, C2_tilde]


def loss(DS, L1, L2, L1_inv, L2_inv, params, loss_type):
    # Convert Matrices to torch tensors
    if isinstance(DS, np.ndarray):
        DS = torch.from_numpy(DS.astype(np.double))
    L1 = torch.from_numpy(L1.astype(np.double))
    L2 = torch.from_numpy(L2.astype(np.double))
    L1_inv = torch.from_numpy(L1_inv.astype(np.double))
    L2_inv = torch.from_numpy(L2_inv.astype(np.double))

    if loss_type == 'w':
        [C1_tilde, C2_tilde] = params
        loss_c = torch.trace(L1_inv) + torch.trace(torch.transpose(DS, 0, 1) @ L2_inv @ DS)
        # svd version
        u, sigma, v = torch.svd(C2_tilde @ DS @ C1_tilde)
        cost = loss_c - 2 * torch.sum(sigma) #torch.abs(sigma))
    # elif loss_type == 'kl':
    #     yy = torch.transpose(DS, 0, 1) @ L2 @ DS
    #     term1 = torch.trace(torch.inverse(L1) @ yy)
    #     K = L1.shape[0]
    #     term2 = torch.logdet(L1) - torch.logdet(yy)
    #     cost = 0.5*(term1 - K + term2)
    elif loss_type == 'l2':
        cost = torch.sum((DS.T @ L2 @ DS - L1) ** 2, dim=1).sum()
    elif loss_type == 'l2-inv':
        cost = torch.sum((DS.T @ L2_inv @ DS - L1_inv) ** 2, dim=1).sum()
    else:
        raise ValueError("loss_type must be 'w', 'l2' or 'l2-inv'.")
    return cost


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
