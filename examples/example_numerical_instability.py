from sinkhorn_663 import sinkhorn
from skh_cpp import sinkhorn_cpp
from sinkhorn_663 import sample_to_prob_vec
import numpy as np
import matplotlib.pyplot as plt

# set up
maxiter = 10000
tol = 1e-6
lamda = np.linspace(120, 180, 61)
skh_res = np.zeros(61)
log_skh_res = np.zeros(61)
test_samp1 = np.random.beta(a=2, b=5, size=2000)
test_samp2 = np.random.uniform(low=5, high=6, size=2000)
M, r, c = sample_to_prob_vec(test_samp1, test_samp2, sigma=0)
# compare
for i in range(61):
    skh_res[i] = sinkhorn_cpp(r, c, M, lamda[i], tol, maxiter)[0]
    log_skh_res[i] = sinkhorn(r, c, M, lamda[i], tol, maxiter, log_domain=True)[0]
plt.figure(figsize=(14, 8))
plt.plot(lamda, skh_res, label="Sinkhorn")
plt.plot(lamda, log_skh_res, label="log-domain Sinkhorn")
plt.xlabel("$\lambda$", fontsize=18)
plt.ylabel("Sinkhorn result", fontsize=16)
plt.title("Sinkhorn and log-domain Sinkhorn results vs $\lambda$", fontsize=16)
plt.legend(fontsize=16)
plt.savefig('report/instability.png')
plt.close()
