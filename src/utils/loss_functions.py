import numpy as np
import scipy.linalg as slg

from utils.help_functions import regularise_and_invert


def l2_loss(x, y, P):
    """Compute the L2 distance between x and P.T @ y @ P"""
    return np.sum((y @ P - P @ x)**2, axis=1).sum()


def l2_inv_loss(x, y, P):
    """Compute the L2 distance between x_inv and P.T @ y_inv @ P"""
    x_inv, y_inv = regularise_and_invert(x, y, alpha=0.1, ones=True)
    return np.sum((y_inv @ P - P @ x_inv)**2, axis=1).sum()


def w2_loss(x, y, P, alpha=0.1, ones=True):
    x_inv, y_inv = regularise_and_invert(x, y, alpha=alpha, ones=ones)
    A = x_inv
    B = P.T @ y_inv @ P
    Root_1 = slg.sqrtm(A)
    result = np.trace(A) + np.trace(B) - 2 * np.trace(slg.sqrtm(Root_1 @ B @ Root_1))
    return result.real
