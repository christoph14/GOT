import functools

import numpy as np
import pygmtools as pygm
from scipy.linalg import pinv


def integer_projected_fixed_point(L1, L2, affinity='default'):
    """Integer Projected Fixed Point from [1]

    Parameters
    ----------
    L1 : array-like of shape (n1,n1)
        Laplacian matrix of the first graph.
    L2 : array-like of shape (n2,n2)
        Laplacian matrix of the first graph.
    affinity: str, default='default'
        The used affinity score.
    Returns
    -------
    X : np.ndarray of shape () TODO add shape
        The computed permutation.

    References
    ----------
    [1] Leordeanu, Marius and Hebert, Martial and Sukthankar, Rahul
        "An Integer Projected Fixed Point Method for Graph Matching and MAP Inference."
        Advances in Neural Information Processing Systems 22, 2009.

    """
    n1 = len(L1)
    n2 = len(L2)
    if affinity == 'default':
        A1 = -L1.copy()
        np.fill_diagonal(A1, 0)
        A2 = -L2.copy()
        np.fill_diagonal(A2, 0)

        conn1, edge1 = pygm.utils.dense_to_sparse(A1)
        conn2, edge2 = pygm.utils.dense_to_sparse(A2)
        gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1)  # set affinity function
        K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, [n1], None, [n2], None,
                                     edge_aff_fn=gaussian_aff)
    elif affinity == 'got':
        # K = np.kron(pinv(L1), pinv(L2))
        conn1, edge1 = pygm.utils.dense_to_sparse(pinv(L1))
        conn2, edge2 = pygm.utils.dense_to_sparse(pinv(L2))
        gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1)  # set affinity function
        K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, [n1], None, [n2], None,
                                     edge_aff_fn=gaussian_aff)
    else:
        raise ValueError("'affinity' must be either 'default' or 'got'.")
    X = pygm.ipfp(K, n1, n2)
    # Hungarian? X = pygm.hungarian(X)
    return X
