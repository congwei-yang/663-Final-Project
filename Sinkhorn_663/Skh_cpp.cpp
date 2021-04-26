#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <vector>
namespace py = pybind11;

Eigen::VectorXd Sinkhorn_cpp(Eigen::VectorXd r, Eigen::MatrixXd C, Eigen::MatrixXd M, double lamda, double tol, int maxiter){
    /**
    *sinkhorn function in cpp with eigen.
    *param r: source empirical measure
    *param C: Target empirical measures. C has the form of a matrix, with columns being target empirical measures
    *param M: Cost matrix
    *param lamda: Entropy regularization parameter
    *param tol: Accuracy tolerance
    *param maxiter: Maximum number of iterations
    */
    Eigen::MatrixXd K = exp(-lamda * M.array());
    Eigen::MatrixXd KT = K.transpose();
    int N = C.cols();
    Eigen::MatrixXd u = (Eigen::MatrixXd::Zero(r.size(), N).array() + 1) / r.size();
    Eigen::VectorXd one_d_r = 1/r.array();
    Eigen::MatrixXd K_tilde = (one_d_r.asDiagonal()) * K;
    Eigen::VectorXd d_prev = Eigen::MatrixXd::Zero(N, 1);
    Eigen::VectorXd d = Eigen::MatrixXd::Zero(N, 1).array() + 2;
    int niter = 0;
    for (int i=0; i<maxiter; i++){
        u = 1/(K_tilde * (C.array()/(KT * u).array()).matrix()).array();
        Eigen::MatrixXd v = C.array() / (KT * u).array();
        d_prev = d;
        Eigen::MatrixXd d_mat = u.array()*((K.array()*M.array()).matrix()*v).array();
        d = d_mat.colwise().sum();
        niter = niter + 1;
        if (((d - d_prev).array().abs()).maxCoeff() <= tol){
          break;
        }
    }
    return d;
}


PYBIND11_MODULE(Skh_cpp, m) {
    m.doc() = "Sinkhorn cpp function";
    m.def("Sinkhorn_cpp", &Sinkhorn_cpp, "A Sinkhorn function");
}
