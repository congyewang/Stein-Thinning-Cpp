//
// Created by congye on 5/11/23.
//

#include <armadillo>
#include "kernel/kernel.h"

int main()
{
    const int dim = 3;
    const int n = 5;
    arma::mat::fixed<n, dim> x(arma::fill::randn);
    arma::mat sx = -x;
    arma::vec::fixed<dim> x_map(arma::fill::randn);
    arma::mat::fixed<dim, dim> linv(arma::fill::eye);
    float s = 3.0;
    float beta = 0.5;

    arma::mat::fixed<n, n> res_kmat_kgm;
    arma::mat::fixed<n, n> res_kmat_imq;

    linv = make_precon(x, sx, "id");

    res_kmat_imq = kmat(x, sx, linv, vfk0_imq, beta);
    res_kmat_imq.print("res_kmat_kgm");

    res_kmat_kgm = kmat(x, sx, x_map, linv, vfk0_centkgm, s, beta);
    res_kmat_kgm.print("res_kmat_kgm");

    auto vfk0_imq = make_imq(x, sx, "id");
    auto vfk0_centkgm = make_centkgm(x, sx, "id");

    // // 准备调用返回的函数的参数
    arma::vec a = arma::randu<arma::vec>(dim);
    arma::vec b = arma::randu<arma::vec>(dim);
    arma::vec sa = arma::randu<arma::vec>(dim);
    arma::vec sb = arma::randu<arma::vec>(dim);

    float res_imq = vfk0_imq(a, b, sa, sb, beta);
    std::cout << "res_imq: " << res_imq << std::endl;

    float res_centkgm = vfk0_centkgm(a, b, sa, sb, x_map, beta);
    std::cout << "res_centkgm: " << res_centkgm << std::endl;

    return 0;
}