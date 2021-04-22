from Sinkhorn_v01 import Sinkhorn_numba
from log_domain_skh import log_domain_sinkhorn_2
import numpy as np

def Sinkhorn(r, C, M, lamda=20, tol=1e-6, maxiter=50, log_domain=False, cpp=False):
    if C.ndim == 1:
        C = C.reshape((-1, 1))
    if log_domain == True:
        N = np.shape(C)[1]
        results = np.zeros(N)
        for i in range(N):
            results[i] = log_domain_sinkhorn_2(r, C[:, i], M, lamda, tol, maxiter)
        return results
    elif cpp == True:
        import cppimport
        Skh_cpp = cppimport.imp("Skh_cpp")
        return Skh_cpp.Sinkhorn_cpp(r, C, M, lamda, tol, maxiter)
    else:
        return Sinkhorn_numba(r, C, M, lamda, tol, maxiter)

