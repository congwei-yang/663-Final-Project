from numpy import zeros, reshape, log, exp
from numpy import sum as npsum
from numba import jit

@jit(nopython = True, parallel = True)
def row_softmin(A, lam):
    """
    Computes row softmin of a matrix
    :param A: Matrix to compute row softmin
    :param lam: Regularization parameter
    :return: Row softmin of a matrix
    """
    epsilon = 1/lam
    result = - epsilon * log(npsum(exp(-A/epsilon), axis = 1))
    return result

@jit(nopython = True, parallel = True)
def col_softmin(A, lam):
    """
    Computes the column softmin of a matrix
    :param A: Matrix given
    :param lam: Regularization parameter
    :return: Column softmin
    """
    epsilon = 1/lam
    result = - epsilon * log(npsum(exp(-A/epsilon), axis = 0))
    return result

@jit(nopython = True, parallel = True)
def log_domain_sinkhorn(r, c, M, lam, tol = 1e-6, maxiter = 10000):
    """
    Computes Sinkhorn distance between empirical measure r and c in log domain
    :param r: Source empirical measure
    :param c: Target empirical measures
    :param M: Cost matrix
    :param lam: Regularization parameter
    :param tol: Accuracy tolerance
    :param maxiter: Maximum iteration number
    :return: An array of Sinkhorn distance
    """
    d = len(r)
    epsilon = 1/lam
    f_prev = zeros(d)
    g_prev = zeros(d)
    f = row_softmin(M - reshape(f_prev, (d,1)) - g_prev, lam = lam) + f_prev + epsilon * log(r)
    g = col_softmin(M - reshape(f, (d,1)) - g_prev, lam = lam) + g_prev + epsilon * log(c)
    dist_prev = 0
    dist = 10
    iteration = 0
    while abs(dist - dist_prev) > tol:
        f_prev = f
        g_prev = g
        f = row_softmin(M - reshape(f_prev, (d,1)) - g_prev, lam = lam) + f_prev + epsilon * log(r)
        g = col_softmin(M - reshape(f, (d,1)) - g_prev, lam = lam) + g_prev + epsilon * log(c)
        K_lg = -lam * M
        u_lg = f / epsilon
        v_lg = g / epsilon
        P_lg = K_lg + reshape(u_lg, (d, 1)) + v_lg
        P = exp(P_lg)
        dist_prev = dist
        dist = npsum(P*M)
        iteration += 1
        if iteration >= maxiter:
            break
    return dist, iteration