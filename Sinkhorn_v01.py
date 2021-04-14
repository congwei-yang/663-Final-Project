import numpy as np
import numba
import idx2numpy
from numba import jit
@jit(nopython=True)
def Sinkhorn(r, C, M, lamda, tol = 1e-7):
    M = M[r > 0]
    r = r[r > 0]
    K = np.exp(-lamda * M)
    N = np.shape(C)[1]
    u = np.ones((len(r), N)) / len(r)
    K_tilde = np.diag(1/r) @ K
    d_prev = np.repeat(2, N) + 0.5
    d = np.ones(N) + 0.5
    niter = 0
    while np.max(np.abs(d - d_prev)) > tol:
        u = 1/(K_tilde @ (C / (K.T @ u)))
        v = C/(K.T @ u)
        d_prev = d
        d = np.sum(u * ((K * M) @ v), axis = 0)
        niter = niter + 1
    return d[0], niter