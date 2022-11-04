import numpy as np
import scipy.linalg as slg

from utils.help_functions import regularise_and_invert


def l2_loss(x, y, P):
    """Compute the Frobeniusnorm of x - P.T @ y @ P"""
    return np.linalg.norm(x - P.T @ y @ P, ord='fro')


def l2_inv_loss(x, y, P, alpha=0.1, ones=True):
    """Compute the Frobeniusnorm of x_inv - P.T @ y_inv @ P"""
    x_inv, y_inv = regularise_and_invert(x, y, alpha, ones)
    return np.linalg.norm(x_inv - P.T @ y_inv @ P, ord='fro')


def w2_loss(x, y, P, alpha=0.1, ones=True):
    x_inv, y_inv = regularise_and_invert(x, y, alpha=alpha, ones=ones)
    A = x_inv
    B = P.T @ y_inv @ P
    Root_1 = slg.sqrtm(A)
    result = np.trace(A) + np.trace(B) - 2 * np.trace(slg.sqrtm(Root_1 @ B @ Root_1))
    return result.real
