from .Sinkhorn_numba import Sinkhorn_numba, Sinkhorn_numba_parallel
from .log_domain_skh import log_domain_sinkhorn_2
from Skh_cpp import Sinkhorn_cpp
import numpy as np

def Sinkhorn(r, C, M, lamda=20, tol=1e-6, maxiter=50, log_domain=False, parallel=False):
    """
    A main sinkhorn function to call appropriate sinkhorn function according to user input
    :param r: Source empirical measure
    :param C: Target empirical measures, C has a form of matrix with its columns being target empirical measures.
    :param M: Cost matrix
    :param lamda: Entropy regularization parameter
    :param tol: Accuracy tolerance
    :param maxiter: Maximum iteration number
    :param log_domain: If True, the function will use log-domain method to compute sinkhorn distance
    :param parallel: If True, the function will use parallel numba version to compute sinkhorn distance
    :return: An array of Sinkhorn distances, number of iteration taken
    """
    if C.ndim == 1:
        C = C.reshape((-1, 1))
    if log_domain == True:
        N = np.shape(C)[1]
        results = np.zeros(N)
        for i in range(N):
            results[i] = log_domain_sinkhorn_2(r, C[:, i], M, lamda, tol, maxiter)[0]
        return results
    elif parallel == True:
        return Sinkhorn_numba_parallel(r, C, M, lamda, tol, maxiter)
    else:
        return Sinkhorn_numba(r, C, M, lamda, tol, maxiter)
