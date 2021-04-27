from sinkhorn_663.image import flatten
import numpy as np

# simulated image data matrix
pixel = 28
img_num = 10
sim_img_mat = [np.random.rand(pixel, pixel) for i in range(img_num)]


def test_flatten():
    # right length
    sim_img_mat_flatten = flatten(sim_img_mat)
    calculated1 = len(sim_img_mat_flatten)
    np.testing.assert_equal(img_num, calculated1)
    # sum to 1
    calculated2 = [sum(sim_img_mat_flatten[i]) for i in range(img_num)]
    np.testing.assert_almost_equal(np.ones(img_num), calculated2)
