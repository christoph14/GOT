import numpy as np
import scipy.linalg as slg

from fGOT import fgot_mgd, fgot_stochastic_mgd


def PstoH(L1, L2, it=10, tau=1):
    """$g(L) = L^{\dagger/2}$ stochastic"""
    return fgot_stochastic_mgd.fgot_stochastic(get_filters(L2, 'got'), get_filters(L1, 'got'), it=it, tau=tau,
                                               n_samples=5, epochs=1000, lr=50*len(L1)*len(L2), std_init=5,
                                               loss_type='w_simple', tol=1e-12, adapt_lr=True)

def P_nv2(L1, L2, it=10, tau=1):
    """$g(L) = L^2$ stochastic"""
    return fgot_stochastic_mgd.fgot_stochastic(get_filters(L2, 'sq'), get_filters(L1, 'sq'), it=it, tau=tau,
                                               n_samples=5, epochs=1000, lr=50*len(L1)*len(L2), std_init=5,
                                               loss_type='w_simple', tol=1e-12, adapt_lr=True)

def find_trace_sink_wass_filters_reg(L1, L2, epsilon=7e-4, method='got', tau=0.2, max_iter=1000):
    n = len(L1)
    m = len(L2)
    p = np.repeat(1/n, n)
    q = np.repeat(1/m, m)
    g1= get_filters(L1, method, tau)
    g2= get_filters(L2, method, tau)

    # Scale epsilon
    epsilon = epsilon * (np.max(g1) * np.max(g2)) / np.sqrt(n * m)

    gw = fgot_mgd.fgot(g1, g2, p, q, epsilon, max_iter=max_iter, tol=1e-9, lapl=True)

    return gw

def get_filters(L1, method, tau=0.2):
    if method == 'got':
        g1 = np.real(slg.sqrtm(fgot_mgd.regularise_invert_one(L1, alpha = 0.1, ones=False )))
    elif method == 'weight':
        g1 = np.diag(np.diag(L1)) - L1
    elif method == 'heat':
        g1 = slg.expm(-tau*L1)
    elif method == 'sqrtL':
        g1 = np.real(slg.sqrtm(L1))
    elif method == 'L':
        g1 = L1
    elif method == 'sq':
        g1 = L1 @ L1
    else:
        raise ValueError("The given filter is not valid.")
    return g1
