import numpy as np


def check_permutation_matrix(P, atol=1e-3):
    P = np.array(P)
    if not P.ndim == 2:
        raise ValueError("P.ndim is not 2.")
    if not P.shape[0] == P.shape[1]:
        raise ValueError("P is not symmetric.")
    if not np.allclose(P.sum(axis=0), 1, atol=atol):
        raise ValueError("Not all columns sum up to 1.")
    if not np.allclose(P.sum(axis=1), 1, atol=atol):
        raise ValueError("Not all rows sum up to 1.")
    if not (np.isclose(P, 0, atol=atol) | np.isclose(P, 1, atol=atol)).all():
        raise ValueError("All entries must be either 0 or 1.")
    return np.round(P).astype(int)


def check_soft_assignment(P, atol=0.001):
    P = np.array(P)
    if not P.ndim == 2:
        raise ValueError("P.ndim is not 2.")
    if not np.allclose(P.sum(axis=0), 1, atol=atol):
        raise ValueError("Not all columns sum up to 1.")
    if not np.allclose(P.sum(axis=1), 1, atol=atol):
        raise ValueError("Not all rows sum up to 1.")
    if not (P >= 0).all():
        raise ValueError("All entries must be non-negative.")
    return P
