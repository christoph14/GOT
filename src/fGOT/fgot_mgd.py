# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as lg

from fGOT.bregman import sinkhorn


#fGOT trace functions
def fgot_grad(C1, C2, T):
    return - 4 * C1 @ T @ C2

def regularise_invert_one(x, alpha, ones):
    if ones:
        x_reg = lg.inv(x   + alpha * np.eye(len(x)) + np.ones([len(x),len(x)])/len(x)) 
    else:
        x_reg = lg.inv(x + alpha * np.eye(len(x)))
    return x_reg

def fgot_loss(tr1, root1, root2, T):
    sumT = np.sum(T)
    return tr1 + np.sum((T @ root2 @ root2 - 2 * root1 @ T @ root2) * T) * len(root1)* len(root2)/(sumT * sumT)
    

def fgot(root1, root2, p, q, epsilon, max_iter=300, tol=1e-9, verbose=False, log=False, lapl = True):

    # Convert matrices, root is g(L)
    root1 = np.asarray(root1, dtype=np.float64)
    root2 = np.asarray(root2, dtype=np.float64)

    # Add entropic regularization
    if lapl:
        root1 = root1 - 0.5*np.diag(np.diag(root1))
        root2 = root2 - 0.5*np.diag(np.diag(root2))    

    # Initialize values
    T = np.outer(p, q)   
    tr1 = np.trace(root1 @ root1)
    cpt = 0
    err = 1

    if log:
        log = {'err': []}

    while err > tol and cpt < max_iter:
        # Evaluate gradient and update T
        Tprev = T
        tens = fgot_grad(root1, root2, T)
        T = sinkhorn(p, q, tens, epsilon)

        # Save log and print loss
        if cpt % 10 == 0:
            err = np.linalg.norm(T - Tprev)
            if log:
                log['err'].append(err)
            if verbose:
                print('GOT loss:', fgot_loss(tr1, root1, root2, T))
        cpt += 1

    if log:
        log['loss'] = fgot_loss(tr1, root1, root2, T)
        return T, log
    else:
        return T
