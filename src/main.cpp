//
// Created by congye on 5/11/23.
//

#include <armadillo>
#include "kernel/kernel.h"

int main() {
    const int dim = 3;
    const int n = 5;
    arma::mat::fixed<n, dim> x(arma::fill::randn);
    arma::mat sx = -x;
    arma::vec::fixed<dim> x_map(arma::fill::randn);
    arma::mat::fixed<dim, dim> linv(arma::fill::eye);
    const float s = 3.0;
    const float beta = 0.5;

    arma::mat::fixed<n, n> res_kmat_kgm;
    arma::mat::fixed<n, n> res_kmat_imq;

    res_kmat_imq = kmat(x, sx, linv, kp_imq, beta);
    res_kmat_imq.print("res_kmat_kgm");

    res_kmat_kgm = kmat(x, sx, x_map, linv, kp_kgm, s, beta);
    res_kmat_kgm.print("res_kmat_kgm");

    return 0;
}