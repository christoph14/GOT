import numpy as np
import torch
torch.set_default_tensor_type('torch.DoubleTensor')
import scipy.linalg as slg
import numpy.linalg as lg


def got_strategy(L1, L2, it, tau, n_samples, epochs, lr, loss_type='w', seed=42, verbose=True, alpha=0.0, ones=True):
    """GOT strategy for graph alignment.

    Parameters
    ----------
    L1 : array-like of shape (n_nodes, n_nodes)
        Laplacian matrix of the first graph.
    L2 : array-like of shape (n_nodes, n_nodes)
        Laplacian matrix of the second graph.
    it : int
        Number of Sinkhorn iterations.
    tau : float
        Sinkhorn parameter.
    n_samples : int
        Number of samples per iteration used by the algorithm.
    epochs : int
        Number of epochs used by the algorithm.
    lr : float
        Learning rate.
    loss_type : str, default='w'
        The loss function to be optimized. Must be in ['w', 'l2'].
    seed : int
        Random seed.
    verbose : bool, default=True
    alpha : float, default=0.0
        Regularization of the Laplacian matrices.
    ones : bool, default=True
        If true, add ones to the matrices before inverting. Does not change the result of GOT error.

    Returns
    -------
    P : np.ndarray of shape (n_nodes, n_nodes)
        The computed permutation matrix.
    """
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
        for sample in range(n_samples):
            # Sampling
            eps = torch.randn(n, n)
            P_noisy = mean + std * eps  # torch.log(1+torch.exp(std)) * eps

            # Cost function
            DS = doubly_stochastic(P_noisy, tau, it)
            cost = cost + loss(DS, L1, L2, L1_inv, L2_inv, params, loss_type)
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


def wasserstein_initialisation(A, B):
    # Wasserstein directly on covariance
    Root_1 = slg.sqrtm(A)
    Root_2 = slg.sqrtm(B)
    C1_tilde = torch.from_numpy(Root_1.astype(np.double))
    C2_tilde = torch.from_numpy(Root_2.astype(np.double))
    return [C1_tilde, C2_tilde]


def doubly_stochastic(P, tau, it):
    """Uses logsumexp for numerical stability."""

    A = P / tau
    for i in range(it):
        A = A - A.logsumexp(dim=1, keepdim=True)
        A = A - A.logsumexp(dim=0, keepdim=True)
    return torch.exp(A)


def regularise_and_invert(x, y, alpha, ones):
    x_reg = regularise_invert_one(x, alpha, ones)
    y_reg = regularise_invert_one(y, alpha, ones)
    return [x_reg, y_reg]


def regularise_invert_one(x, alpha, ones):
    if ones:
        x_reg = lg.inv(x   + alpha * np.eye(len(x)) + np.ones([len(x),len(x)])/len(x))
    else:
        x_reg = lg.pinv(x + alpha * np.eye(len(x)))
    return x_reg


def loss(DS, L1, L2, L1_inv, L2_inv, params, loss_type):
    # Convert matrices to torch tensors
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
        sigma = torch.linalg.svdvals(C2_tilde @ DS @ C1_tilde)
        cost = loss_c - 2 * torch.sum(sigma) #torch.abs(sigma))
    elif loss_type == 'l2':
        cost = torch.sum((DS.T @ L2 @ DS - L1) ** 2, dim=1).sum()
    elif loss_type == 'l2-inv':
        cost = torch.sum((DS.T @ L2_inv @ DS - L1_inv) ** 2, dim=1).sum()
    else:
        raise ValueError("loss_type must be 'w', 'l2' or 'l2-inv'.")
    return cost
