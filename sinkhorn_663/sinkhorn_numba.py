import numpy as np
from numba import jit, prange
@jit(nopython=True)
def sinkhorn_numba(r, C, M, lamda, tol = 1e-6, maxiter = 10000):
    """
    Numba version of Sinkhorn function
    :param r: Source empirical measure
    :param C: Target empirical measures. C has a format of a matrix, with columns being target empirical measures
    :param M: Cost matrix
    :param lamda: Entropy regularization parameter
    :param tol: Accuracy tolerance
    :param maxiter: Maximum number of iterations
    :return: An array of sinkhorn distances, number of iteration taken
    """
    M = M[r > 0]
    r = r[r > 0]
    K = np.exp(-lamda * M)
    N = np.shape(C)[1]
    u = np.ones((len(r), N)) / len(r)
    K_tilde = np.diag(1/r) @ K
    d_prev = np.repeat(2., N)
    d = np.ones(N) + 0.5
    for i in range(maxiter):
        u_new = 1/(K_tilde @ (C / (K.T @ u)))
        if np.max(np.abs(u_new - u)) <= tol:
            break
        u = u_new
    v = C/(K.T @ u)
    d = np.sum(u * ((K * M) @ v), axis = 0)
    return d[0], i

@jit(nopython=True, parallel = True)
def sinkhorn_numba_parallel(r, C, M, lamda, tol = 1e-6, maxiter = 10000):
    """
    Numba version of Sinkhorn function with numba parallel
    :param r: Source empirical measure
    :param C: Target empirical measures. C has a format of a matrix, with columns being target empirical measures
    :param M: Cost matrix
    :param lamda: Entropy regularization parameter
    :param tol: Accuracy tolerance
    :param maxiter: Maximum number of iterations
    :return: An array of sinkhorn distances, number of iteration taken
    """
    M = M[r > 0]
    r = r[r > 0]
    K = np.exp(-lamda * M)
    N = np.shape(C)[1]
    u = np.ones((len(r), N)) / len(r)
    K_tilde = np.diag(1/r) @ K
    d_prev = np.repeat(2., N)
    d = np.ones(N) + 0.5
    for i in prange(maxiter):
        u_new = 1/(K_tilde @ (C / (K.T @ u)))
        if np.max(np.abs(u_new - u)) <= tol:
            break
        u = u_new
    v = C/(K.T @ u)
    d = np.sum(u * ((K * M) @ v), axis = 0)
    return d[0], i
