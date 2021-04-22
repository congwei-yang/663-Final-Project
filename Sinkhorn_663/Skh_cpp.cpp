<%
cfg['compiler_args'] = ['-std=c++11']
cfg['include_dirs'] = ['./eigen-3.3.9']
setup_pybind11(cfg)
%>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <vector>
namespace py = pybind11;

Eigen::VectorXd Sinkhorn_cpp(Eigen::VectorXd r, Eigen::MatrixXd C, Eigen::MatrixXd M, double lamda, double tol, int maxiter){
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
    }
    return d;
}


PYBIND11_MODULE(Skh_cpp, m) {
    m.doc() = "Sinkhorn cpp function";
    m.def("Sinkhorn_cpp", &Sinkhorn_cpp, "A Sinkhorn function");
}