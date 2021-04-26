import numpy as np


def cost_mat(d):
    """
    Compute the cost matrix for `d` dimension image matrix.
    Args
        d: the dimension of the image matrix
    Returns
        cost matrix for the image matrix. This can be an input `M` in sinkhorn functions.
    """
    M = np.zeros((d**2, d**2))
    for i in range(1, d**2 + 1):
        for j in range(1, d**2 + 1):
            x1 = i//d + 1
            y1 = -(i%d)
            if y1 == 0:
                y1 = -d
                x1 -= 1
            x2 = j//d + 1
            y2 = -(j%d)
            if y2 == 0:
                y2 = -d
                x2 -= 1
            M[i-1][j-1] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return (M)
