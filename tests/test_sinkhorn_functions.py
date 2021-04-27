from sinkhorn_663 import sinkhorn, log_domain_sinkhorn, sinkhorn_numba, sinkhorn_numba_parallel
from skh_cpp import sinkhorn_cpp
from sinkhorn_663 import sample_to_prob_vec
import numpy as np

# two simulated data iid from the same distribution
N = 3000
np.random.seed(1)
u1 = np.random.beta(a=1, b=2, size=N)
v1 = np.random.beta(a=1, b=2, size=N)
M1, r1, c1 = sample_to_prob_vec(u1, v1)
c1 = c1.reshape(-1, 1)

# two simulated data from two distributions with a known OT distance
np.random.seed(1)
u2 = np.random.uniform(0, 1, size=N)
v2 = np.random.uniform(10, 11, size=N)
M2, r2, c2 = sample_to_prob_vec(u2, v2, 1)
c2 = c2.reshape(-1, 1)

# set parameters
maxiter = 10000
tol = 1e-6
lamda = 20


def test_sinkhorn():
    true1 = 0
    calculated1 = sinkhorn(r1, c1, M1, lamda, tol, maxiter)[0]
    np.testing.assert_almost_equal(true1, calculated1, decimal=1)

    true2 = 10
    calculated2 = sinkhorn(r2, c2, M2, lamda, tol, maxiter)[0]
    np.testing.assert_almost_equal(true2, calculated2, decimal=1)


def test_log_domain_sinkhorn():
    true1 = 0
    calculated1 = log_domain_sinkhorn(r1, c1[:, 0], M1, lamda, tol, maxiter)[0]
    np.testing.assert_almost_equal(true1, calculated1, decimal=1)

    true2 = 10
    calculated2 = log_domain_sinkhorn(r2, c2[:, 0], M2, lamda, tol, maxiter)[0]
    np.testing.assert_almost_equal(true2, calculated2, decimal=1)


def test_sinkhorn_numba():
    true1 = 0
    calculated1 = sinkhorn_numba(r1, c1, M1, lamda, tol, maxiter)[0]
    np.testing.assert_almost_equal(true1, calculated1, decimal=1)

    true2 = 10
    calculated2 = sinkhorn_numba(r2, c2, M2, lamda, tol, maxiter)[0]
    np.testing.assert_almost_equal(true2, calculated2, decimal=1)


def test_sinkhorn_numba_parallel():
    true1 = 0
    calculated1 = sinkhorn_numba_parallel(r1, c1, M1, lamda, tol, maxiter)[0]
    np.testing.assert_almost_equal(true1, calculated1, decimal=1)

    true2 = 10
    calculated2 = sinkhorn_numba_parallel(r2, c2, M2, lamda, tol, maxiter)[0]
    np.testing.assert_almost_equal(true2, calculated2, decimal=1)


def test_sinkhorn_cpp():
    true1 = 0
    calculated1 = sinkhorn_cpp(r1, c1, M1, lamda, tol, maxiter)[0]
    np.testing.assert_almost_equal(true1, calculated1, decimal=1)

    true2 = 10
    calculated2 = sinkhorn_cpp(r2, c2, M2, lamda, tol, maxiter)[0]
    np.testing.assert_almost_equal(true2, calculated2, decimal=1)
