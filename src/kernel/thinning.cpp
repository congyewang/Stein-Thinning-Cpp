#include <armadillo>
#include "kernel.h"

arma::uvec thin(const arma::mat &smp, const arma::mat &scr, const int m, const bool stnd = true,  const bool verb = false)
{
    int n = smp.n_rows;
    int d = smp.n_cols;

    arma::mat smp_copy = smp;
    arma::mat scr_copy = scr;

    if (stnd == true)
    {
        arma::rowvec loc = arma::mean(smp_copy, 0);
        arma::rowvec scl = arma::mean(arma::abs(smp_copy.each_row() - loc), 0);
        smp_copy = smp_copy.each_row() / scl;
        scr_copy = scr_copy.each_row() % scl;
    }

    // Pre-allocate arrays
    arma::mat k0(n, m);
    arma::uvec idx(m, arma::fill::zeros);

    // // Populate columns of k0 as new points are selected
    k0.col(0) = vectorised_stein_kernel_imq(smp_copy, smp_copy, scr_copy, scr_copy);
    idx.row(0) = k0.col(0).index_min();
    if (verb == true) std::cout << "THIN: 1 of " << m << std::endl;

    for (int i = 1; i < m; i++)
    {
        arma::mat smp_last = arma::repelem(smp_copy.row(idx.row(i - 1)[0]), n, 1);
        arma::mat scr_last = arma::repelem(scr_copy.row(idx.row(i - 1)[0]), n, 1);
        k0.col(i) = vectorised_stein_kernel_imq(smp_copy, smp_last, scr_copy, scr_last);

        idx.row(i) = (k0.col(0) + 2 * arma::sum(k0.cols(1, i), 1)).index_min();
        if (verb == true) std::cout << "THIN: " << i + 1 << " of " << m << std::endl;
    }

    return idx;
}
