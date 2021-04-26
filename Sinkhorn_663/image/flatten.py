import numpy as np


def flatten(data):
    """
    Flatten a list of image data from matrices to vectors.
    :param data: A list of imate matrices
    :return: A list of image vectors, which can be put into sinkhorn functions
    """
    data_flat = []
    for i in range(0, len(data)):
        K = data[i]
        K_normal = K / sum(np.matrix.flatten(K, order='F'))
        K_flat = np.matrix.flatten(K_normal, order='F')
        data_flat.append(K_flat)
    return data_flat
