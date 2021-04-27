from sinkhorn_663 import sinkhorn, log_domain_sinkhorn, sinkhorn_numba, sinkhorn_numba_parallel
from skh_cpp import sinkhorn_cpp
from sinkhorn_663 import sample_to_prob_vec
import numpy as np


# create simulation data
N = 3000
np.random.seed(1)
u2 = np.random.uniform(0, 1, size = N)
v2 = np.random.uniform(10, 11, size = N)
M2, r2, c2 = sample_to_prob_vec(u2, v2, 1)
c2 = c2.reshape(-1, 1)
# set parameters
maxiter = 10000
tol = 1e-6
lamda = 20

print("input\n************************")
print("Source empirical measure:\n", r2)
print("Target empirical measures:\n", c2)
print("Cost matrix:\n", M2)
print("Maximum iteration number: ", maxiter)
print("Accuracy tolerance: ", tol)
print("Entropy regularization parameter: ", lamda)

# use default sinkhorn
dist1, iter1 = sinkhorn(r2, c2, M2, lamda, tol, maxiter)
# results print
print("\noutput: default sinkhorn\n************************")
print("Sinkhorn distances: ", dist1)
print("number of iteration taken: ", iter1)

# use paralyzed sinkhorn
dist2, iter2 = sinkhorn(r2, c2, M2, lamda, tol, maxiter, parallel=True)
# results print
print("\noutput: paralyzed sinkhorn\n************************")
print("Sinkhorn distances: ", dist2)
print("number of iteration taken: ", iter2)

# use log_domain sinkhorn
dist3 = sinkhorn(r2, c2, M2, lamda, tol, maxiter, log_domain=True)
# results print
print("\noutput: log_domain sinkhorn\n************************")
print("Sinkhorn distances: ", dist3)

# use cpp sinkhorn
dist4 = sinkhorn_cpp(r2, c2, M2, lamda, tol, maxiter)
# results print
print("\noutput: log_domain sinkhorn\n************************")
print("Sinkhorn distances: ", dist4)
