def remove_zeros(r, M):
    """
    image processor to remove zero
    :param r: Source image measure
    :param M: Cost matrix
    :return: Processed r and M with zeros removed
    """
    M = M[r > 0]
    r = r[r > 0]
    return r, M