import networkx as nx
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


def w2_loss(x, y, P, alpha=0, ones=True):
    A, B = regularise_and_invert(x, P.T @ y @ P, alpha=alpha, ones=ones)
    root = slg.sqrtm(A)
    result = np.trace(A) + np.trace(B) - 2 * np.trace(slg.sqrtm(root @ B @ root))
    return result.real


def gw_loss(G1, G2, T, p=None, q=None, atol=0.01):
    """Compute the GW loss for graphs G1, G2 and coupling T between p and q"""
    # Compute the shortest path matrices
    C1 = nx.floyd_warshall_numpy(G1)
    C2 = nx.floyd_warshall_numpy(G2)

    # Set default distributions in none are given
    n1 = len(C1)
    n2 = len(C2)
    if p is None:
        p = np.full(n1, 1/n1)
    if q is None:
        q = np.full(n2, 1/n2)

    # Check coupling
    if not np.allclose(np.sum(T, axis=1), p, atol=atol):
        print(np.sum(T, axis=1))
        print(p)
        raise ValueError("The given coupling is not valid.")
    if not np.allclose(np.sum(T, axis=0), q, atol=atol):
        print(np.sum(T, axis=0))
        print(q)
        raise ValueError("The given coupling is not valid.")
    return np.sum([[[[(C1[i,k] - C2[j,l])**2 * T[i,j] * T[k,l]
                      for l in range(n2)]
                     for k in range(n1)]
                    for j in range(n2)]
                   for i in range(n1)])
