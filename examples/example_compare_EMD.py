from sinkhorn_663 import sinkhorn
from sinkhorn_663.image import cost_mat, flatten, remove_zeros
from skh_cpp import sinkhorn_cpp
import numpy as np
import matplotlib.pyplot as plt

import scipy.io
import pandas as pd
import ot

# read in data
mnist = scipy.io.loadmat('data/mnist.mat')
images = mnist.get('trainX')
# set up
tol = 1e-6
N = 1000
d = np.sqrt(len(images[1, :]))
lams = [1, 5, 10, 20, 50]
EMD = np.zeros((len(lams), N))
SKH = np.zeros((len(lams), N))
# preprocess image
np.random.seed(2)
X1 = np.random.choice(len(images), N)
X2 = np.random.choice(len(images), N)
M = cost_mat(int(d))
p, q = flatten(images[X1, :]), flatten(images[X2, :])
# compare
for i, lam in enumerate(lams):
    for j in range(N):
        r, M_ = remove_zeros(p[j], M)
        SKH[i, j] = sinkhorn_cpp(r, q[j], M_, lam, tol, 5000)[0]
        EMD[i, j] = ot.emd2(p[j], q[j], M)
df = pd.DataFrame(((SKH - EMD) / EMD).T, columns=['1', '5', '10', '20', '50'])
compare_boxplot = plt.figure()
df.boxplot()
compare_boxplot.savefig("report/emd_deviation.png", format="png")
plt.close()
