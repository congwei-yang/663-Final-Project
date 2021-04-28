# sinkhorn_663: a Python package for Implementation and Optimization of Sinkhorn Algorithm

The Sinkhorn algorithm is proposed by M. Cuturi in 2013, which provides an efficient approximation to the optimal transport (OT) distance. We built the sinkhorn implementation package `sinkhorn_663` and incorporated numba and c++ to optimize the Sinkhorn function. We also provided a few additional module for the user to conveniently convert random samples or images into empirical measures for Sinkhorn computation.

# Installation

Use the following command in the terminal to install the package: 

`pip install --index-url https://test.pypi.org/simple/ sinkhorn_663`

Note that the package requires the up-to-date pybind11(2.6.2). Please install or update the pybind11 before installing the sinkhorn package. 

To import all the modules and functions, use the following code: 

`from sinkhorn_663 import sinkhorn, log_domain_sinkhorn, sinkhorn_numba, sinkhorn_numba_parallel`

`from sinkhorn_663 import sample_to_prob_vec, sample_to_prob_vec_nD`

`from sinkhorn_663.image import cost_mat, flatten, remove_zeros`

`from skh_cpp import sinkhorn_cpp`

# [Eigen](https://github.com/congwei-yang/663-Final-Project/tree/main/sinkhorn_663/Eigen)

For the purpose of optimization, we write the function in c++ and use pybind11 to wrap them as `sinkhorn_cpp`. Note that we use **eigen** library to help us do matrix computation. To make our package function well, we include necessary documents of **eigen** in our package directory, which is everything in `sinkhorn_663/Eigen`. That's why Github shows our repository mostly composed of c++. Also, before uploading our package, we add a `MANIFEST.in` with `recursive-include sinkhorn_663/Eigen *` to claim that the Eigen directory is included in our package.

# [Data](https://github.com/congwei-yang/663-Final-Project/tree/main/data)

In `data/` directory, we store two data sets we use for examples. One is MNIST digits dataset, the other is CalTech 101 Silhouettes Data Set. They are stored in `.mat` format. You can learn how to read in and extract the information from our examples.

# [Examples](https://github.com/congwei-yang/663-Final-Project/tree/main/examples)

In `examples/` directory, we present codes showing how to use each function, real data set and repeat our results.

1) `example_sample_to_prob_vec.py` and `example_sinkhorn_functions.py` are codes showing how to use corresponding functions.
2) `example_compare_EMD.py`, `example_complexity.py` `example_numerical_instability.py`, and `example_data_silhouettes.py` are codes using data and producing results.

# [Tests](https://github.com/congwei-yang/663-Final-Project/tree/main/tests)

In `tests/` directory, we do the following tests:

1) Test different versions of `sinkhorn()` functions with simulated data as sanity check.
2) Test the `sample_to_prob_vec()` returns the right dimensions and probability vector sum up to 1.
3) Test the `cost_mat()` returns the right dimensions.
4) Test the `flatten()` returns the right length of list and each vector in the list sum up to 1.
