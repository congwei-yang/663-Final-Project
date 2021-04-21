import numpy as np
from numpy import zeros, reshape, log, exp
from numpy import sum as npsum
from numba import jit

@jit(nopython = True, parallel = True)
def row_softmin(A, lam):
    epsilon = 1/lam
    result = - epsilon * log(npsum(exp(-A/epsilon), axis = 1))
    return result

@jit(nopython = True, parallel = True)
def col_softmin(A, lam):
    epsilon = 1/lam
    result = - epsilon * log(npsum(exp(-A/epsilon), axis = 0))
    return result

@jit(nopython = True, parallel = True)
def log_domain_sinkhorn_2(lam, M, r, c):
    d = len(r)
    epsilon = 1/lam
    f_prev = zeros(d)
    g_prev = zeros(d)
    f = row_softmin(M - reshape(f_prev, (d,1)) - g_prev, lam = lam) + f_prev + epsilon * log(r)
    g = col_softmin(M - reshape(f, (d,1)) - g_prev, lam = lam) + g_prev + epsilon * log(c)
    dist_prev = 0
    dist = 10
    iteration = 0
    while abs(dist - dist_prev) > 1e-300:
        f_prev = f
        g_prev = g
        f = row_softmin(M - reshape(f_prev, (d,1)) - g_prev, lam = lam) + f_prev + epsilon * log(r)
        g = col_softmin(M - reshape(f, (d,1)) - g_prev, lam = lam) + g_prev + epsilon * log(c)
        iteration += 1
        K_lg = -lam * M
        u_lg = f / epsilon
        v_lg = g / epsilon
        P_lg = K_lg + reshape(u_lg, (d, 1)) + v_lg
        P = exp(P_lg)
        dist_prev = dist
        dist = npsum(P*M)
    return [dist, iteration]