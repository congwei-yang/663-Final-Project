from sinkhorn_663 import sinkhorn
from sinkhorn_663.image import cost_mat, flatten
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd

# read in data
silhouettes = scipy.io.loadmat('data/silhouettes.mat')
img_mat = silhouettes.get('X')  # get image matrix from dictionary
# image samples: nine images for plane, umbrella and cat
pixel = 28
plane_row = [5, 6, 10]
umbrella_row = [8121, 8123, 8124]
cat_row = [8505, 8507, 8515]
compare_img_mat = [img_mat[i, ].reshape(pixel, pixel).T for i in plane_row + umbrella_row + cat_row]
# image name
plane_idx = ['plane' + str(i) for i in range(1, 4)]
umbrella_idx = ['umbrella' + str(i) for i in range(1, 4)]
cat_idx = ['cat' + str(i) for i in range(1, 4)]
all_idx = plane_idx + umbrella_idx + cat_idx
# plot
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(7, 7))
for p in range(9):
    i = p // 3
    j = p % 3
    axes[i, j].imshow(compare_img_mat[p], cmap='gray')
    axes[i, j].axis('off')
    axes[i, j].set_title(all_idx[p])
fig.suptitle('Nine Image Samples from silhouettes data')
plt.savefig('report/silhouettes_image_sample.png')
plt.close()

# use of functions in sinkhorn_663.image
M_img = cost_mat(pixel)
compare_img_flat = flatten(compare_img_mat)
maxiter = 10000
tol = 1e-4
lamda = 20

# use of sinkhorn
compare_result = np.zeros([9, 9])
for i in range(9):
    for j in range(9):
        compare_result[i, j] = sinkhorn(compare_img_flat[i], compare_img_flat[j], M_img, lamda, tol, maxiter)[0]
result_df = pd.DataFrame(compare_result, index=all_idx, columns=all_idx).round(2)
result_df.to_pickle("report/result_df")
print(result_df)