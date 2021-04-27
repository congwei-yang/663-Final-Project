from sinkhorn_663 import sinkhorn
from sinkhorn_663 import sample_to_prob_vec
import numpy as np
import matplotlib.pyplot as plt

# set up
maxiter = 10000
np.random.seed(1)
size = [64, 128, 256, 512, 1024, 2048]
nrep = 10
iter_num_1 = np.zeros((nrep, len(size)))
iter_num_10 = np.zeros((nrep, len(size)))
iter_num_50 = np.zeros((nrep, len(size)))
for j in range(nrep):
    for i in range(6):
        p_samp = np.random.beta(a=2, b=5, size=size[i])
        q_samp = np.random.normal(size=size[i])
        M, r, c = sample_to_prob_vec(p_samp, q_samp, sigma=0)
        iter_num_1[j, i] = sinkhorn(r, c, M, 1, 1e-11, maxiter)[1]
        iter_num_10[j, i] = sinkhorn(r, c, M, 10, 1e-11, maxiter)[1]
        iter_num_50[j, i] = sinkhorn(r, c, M, 50, 1e-11, maxiter)[1]

# mean
mean_iter_1 = np.mean(iter_num_1, axis=0)
mean_iter_10 = np.mean(iter_num_10, axis=0)
mean_iter_50 = np.mean(iter_num_50, axis=0)
# std
std_iter_1 = np.std(iter_num_1, axis=0)
std_iter_10 = np.std(iter_num_10, axis=0)
std_iter_50 = np.std(iter_num_50, axis=0)

plt.figure(figsize=(15, 10))
plt.plot(size, mean_iter_1, label="$\lambda = 1$", marker='o')
plt.fill_between(size, mean_iter_1 - std_iter_1, mean_iter_1 + std_iter_1, alpha=0.2)
plt.plot(size, mean_iter_10, label="$\lambda = 10$", marker='o')
plt.fill_between(size, mean_iter_10 - std_iter_10, mean_iter_10 + std_iter_10, alpha=0.2)
plt.plot(size, mean_iter_50, label="$\lambda = 50$", marker='o')
plt.fill_between(size, mean_iter_50 - std_iter_50, mean_iter_50 + std_iter_50, alpha=0.2)
plt.yscale('log')
plt.xscale('log')
plt.xticks(ticks=size)
plt.xlabel("Histogram size", fontsize=16)
plt.ylabel("Sinkhorn Iteration Number", fontsize=16)
plt.title("Sinkhorn Iteration Number under Different Histogram Size and $\lambda$", fontsize=16)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=16)
plt.savefig('report/empirical_complexity.png')
plt.close()
