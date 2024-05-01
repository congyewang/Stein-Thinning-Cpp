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
    int s = 3;
    float beta = 0.5;

    arma::mat::fixed<n, n> res_kmat_kgm;
    arma::mat::fixed<n, n> res_kmat_imq;
    arma::mat::fixed<n, n> res_kmat_test;

    linv = make_precon(x, sx, "id");

    res_kmat_imq = kmat(x, sx, linv, vfk0_imq, beta);

    res_kmat_kgm = kmat(x, sx, x_map, linv, vfk0_centkgm, s, beta);

    auto stein_kernel_imq = make_imq(x, sx, "id");
    auto stein_kernel_centkgm = make_centkgm(x, sx, "id");

    arma::vec a = arma::randu<arma::vec>(dim);
    arma::vec b = arma::randu<arma::vec>(dim);
    arma::vec sa = arma::randu<arma::vec>(dim);
    arma::vec sb = arma::randu<arma::vec>(dim);

    float res_imq = stein_kernel_imq(a, b, sa, sb);

    float res_centkgm = stein_kernel_centkgm(a, b, sa, sb, x_map);

    res_kmat_test = kmat(x, sx, x_map, stein_kernel_centkgm);
    res_kmat_test.print("res_kmat_test");

    return 0;
}