import numpy as np
import torch
import scipy.linalg as slg

from utils.help_functions import regularise_and_invert

torch.set_default_tensor_type('torch.DoubleTensor')


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
    else:
        raise NotImplementedError("Only strategies 'GOT', 'L2', 'L2-inv' are implemented.")
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
        cost = torch.sum((L2 @ DS - DS @ L1) ** 2, dim=1).sum()
    elif loss_type == 'l2-inv':
        cost = torch.sum((L2_inv @ DS - DS @ L1_inv) ** 2, dim=1).sum()
    else:
        raise ValueError("loss_type must be 'w', 'l2' or 'l2-inv'.")
    return cost
