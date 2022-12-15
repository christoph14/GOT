import numpy as np
import scipy.linalg as slg


def wasserstein_distance(A, B):
    """Calculate the wasserstein distance between Gaussian distributions.

    Parameters
    ----------
    A : array-like of shape (n, n)
        Covariance matrix of the first distribution.
    B : float
        Calculated Wasserstein distance.

    Returns
    -------
    distance : numpy.ndarray of shape (n, n)
        The calculated transportation matrix.
    """
    Root_1 = slg.sqrtm(A)
    result = np.trace(A) + np.trace(B) - 2 * np.trace(slg.sqrtm(Root_1 @ B @ Root_1))
    distance = np.abs(result.real)
    return distance
