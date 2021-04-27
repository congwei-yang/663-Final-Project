from sinkhorn_663 import sample_to_prob_vec
import numpy as np

# create simulation data
N = 3000
np.random.seed(1)
u1 = np.random.beta(a=1, b=2, size=N)
v1 = np.random.beta(a=1, b=2, size=N)
# input print
print("input\n************************")
print("p_sample:\n", u1)
print("q_sample:\n", v1)

# use sample_to_prob_vec
M1, r1, c1 = sample_to_prob_vec(u1, v1)

# results print
print("\noutput\n************************")
print("Distance Matrix:\n", M1)
print("probability measure p_vec:\n", r1)
print("probability measure 1_vec:\n", c1)
