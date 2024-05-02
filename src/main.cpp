#include <armadillo>
#include "kernel/kernel.h"
#include "kernel/thinning.h"

int main()
{
    const int dim = 4;
    const int num = 6;
    arma::mat::fixed<num, dim> x(arma::fill::randn);
    arma::mat sx = -x;
    arma::vec::fixed<dim> x_map(arma::fill::randn);
    arma::mat::fixed<dim, dim> linv(arma::fill::eye);
    int s = 3;
    float beta = 0.5;

    arma::mat::fixed<num, num> res_kmat_kgm;
    arma::mat::fixed<num, num> res_kmat_imq;

    linv = make_precon(x, sx, "id");

    res_kmat_imq = kmat(x, sx, linv, stein_kernel_imq, beta);

    res_kmat_kgm = kmat(x, sx, x_map, linv, stein_kernel_centkgm, s, beta);

    auto stein_kernel_imq = make_imq(x, sx, "id");
    auto stein_kernel_centkgm = make_centkgm(x, sx, "id");

    arma::mat::fixed<num, dim> a(arma::fill::randn);
    arma::mat::fixed<num, dim> b(arma::fill::randn);
    arma::mat::fixed<num, dim> sa(arma::fill::randn);
    arma::mat::fixed<num, dim> sb(arma::fill::randn);

    arma::vec res_vec_centkgm;
    arma::vec res_vec_imq;

    res_vec_centkgm = vectorised_stein_kernel_centkgm(a, b, sa, sb, x_map);

    res_vec_imq = vectorised_stein_kernel_imq(a, b, sa, sb);

    const int n_rand = 2;
    arma::mat::fixed<n_rand, dim> x_new(arma::fill::randn);
    arma::mat sx_new = -x_new;

    arma::vec res_vfps_centkgm;
    arma::vec res_vfps_imq;

    res_vfps_centkgm = vfps(x_new, sx_new, x, sx, x_map, 1, vectorised_stein_kernel_centkgm);
    res_vfps_imq = vfps(x_new, sx_new, x, sx, 1, vectorised_stein_kernel_imq);

    arma::uvec idx = thin(x, sx, 3);
    idx.print("idx: ");

    return 0;
}