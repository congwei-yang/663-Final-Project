import numpy as np


def sample_to_prob_vec(p_sample, q_sample, sigma = 0):
    """
    Convert 1-dimensional samples to probability measures and the corresponding cost matrix. Can add Gaussian smoothness
    :param p_sample: Sample 1
    :param q_sample: Sample 2
    :param sigma: Gaussian smoothness parameter
    :return: Cost matrix cost_mat, probability measure p_vec, probability measure q_vec
    """
    if sigma == 0:
        p_final_sample = p_sample
        q_final_sample = q_sample
    else:
        p_normal_sample = np.random.normal(loc=0, scale=sigma, size=len(p_sample))
        q_normal_sample = np.random.normal(loc=0, scale=sigma, size=len(q_sample))
        p_final_sample = p_sample + p_normal_sample
        q_final_sample = q_sample + q_normal_sample

    p_vec = np.repeat(1 / len(p_final_sample), len(p_final_sample))
    q_vec = np.repeat(1 / len(q_final_sample), len(q_final_sample))

    cost_mat = np.zeros((len(p_vec), len(q_vec)))
    for i in range(0, len(p_vec)):
        cost_mat[i, :] = np.abs(p_final_sample[i] - q_final_sample)
    return cost_mat, p_vec, q_vec


def sample_to_prob_vec_nD(p_sample, q_sample, sigma):
    """
    Convert n-dimensional samples to probability measures and the corresponding cost matrix. Can add Gaussian smoothness
    :param p_sample: Sample 1
    :param q_sample: Sample 2
    :param sigma: Gaussian smoothness parameter
    :return: Cost matrix cost_mat, probability measure p_vec, probability measure q_vec
    """
    if sigma == 0:
        p_final_sample = p_sample
        q_final_sample = q_sample
    else:
        n_p = np.shape(p_sample)[1]
        n_q = np.shape(q_sample)[1]
        p_normal_sample = np.random.multivariate_normal(mean=np.zeros(n_p), cov=np.diag(np.repeat(sigma ** 2, n_p)),
                                                        size=len(p_sample))
        q_normal_sample = np.random.multivariate_normal(mean=np.zeros(n_q), cov=np.diag(np.repeat(sigma ** 2, n_q)),
                                                        size=len(q_sample))
        p_final_sample = p_sample + p_normal_sample
        q_final_sample = q_sample + q_normal_sample

    p_vec = np.repeat(1 / len(p_final_sample), len(p_final_sample))
    q_vec = np.repeat(1 / len(q_final_sample), len(q_final_sample))

    cost_mat = np.zeros((len(p_vec), len(q_vec)))
    for i in range(0, len(p_vec)):
        cost_mat[i, :] = np.sqrt(np.sum((q_final_sample - p_final_sample[i]) ** 2, axis=1))
    return cost_mat, p_vec, q_vec