import numpy as np
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
    root = slg.sqrtm(A)
    result = np.trace(A) + np.trace(B) - 2 * np.trace(slg.sqrtm(root @ B @ root))
    distance = np.abs(result.real)
    return distance
