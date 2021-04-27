from sinkhorn_663.image import cost_mat
import numpy as np


def test_cost_mat():
    d = 18
    test_mat = cost_mat(d)
    # right dimension
    np.testing.assert_equal([d**2, d**2], test_mat.shape)
