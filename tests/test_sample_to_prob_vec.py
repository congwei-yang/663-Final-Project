from sinkhorn_663 import sample_to_prob_vec
import numpy as np

# create simulation data
N = 3000
np.random.seed(1)
u1 = np.random.beta(a=1, b=2, size=N)
v1 = np.random.beta(a=1, b=2, size=N)


def test_sample_to_prob_vec():
    M1, r1, c1 = sample_to_prob_vec(u1, v1)
    # right dimension
    np.testing.assert_equal(N, len(r1))
    np.testing.assert_equal(N, len(c1))
    np.testing.assert_equal([N, N], M1.shape)
    # sum to 1
    np.testing.assert_almost_equal(1, sum(r1))
    np.testing.assert_almost_equal(1, sum(c1))
