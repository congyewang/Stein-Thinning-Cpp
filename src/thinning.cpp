#include <armadillo>
#include <stdexcept>
#include "kernel.h"
#include "thinning.h"

arma::uvec stein_thinning::thin(const arma::mat &smp, const arma::mat &scr, const int m, const std::string &pre, const bool stnd, const bool verb)
{
    int n = smp.n_rows;
    int d = smp.n_cols;

    // Argument checks
    if (n == 0 || d == 0)
    {
        throw std::invalid_argument("smp is empty.");
    }

    if (scr.n_rows != n || scr.n_cols != d)
    {
        throw std::invalid_argument("Dimensions of smp and scr are inconsistent.");
    }

    if (smp.has_nan() || scr.has_nan())
    {
        throw std::invalid_argument("smp or scr contains NaNs.");
    }

    if (smp.has_inf() || scr.has_inf())
    {
        throw std::invalid_argument("smp or scr contains infs.");
    }

    arma::mat smp_copy = smp;
    arma::mat scr_copy = scr;

    // Standardisation
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
    k0.col(0) = stein_thinning::kernel::vectorised_stein_kernel_imq(smp_copy, smp_copy, scr_copy, scr_copy, pre);
    idx.row(0) = k0.col(0).index_min();
    if (verb == true)
        std::cout << "THIN: 1 of " << m << std::endl;

    for (int i = 1; i < m; i++)
    {
        arma::mat smp_last = arma::repelem(smp_copy.row(idx.row(i - 1)[0]), n, 1);
        arma::mat scr_last = arma::repelem(scr_copy.row(idx.row(i - 1)[0]), n, 1);
        k0.col(i) = stein_thinning::kernel::vectorised_stein_kernel_imq(smp_copy, smp_last, scr_copy, scr_last);

        idx.row(i) = (k0.col(0) + 2 * arma::sum(k0.cols(1, i), 1)).index_min();
        if (verb == true)
            std::cout << "THIN: " << i + 1 << " of " << m << std::endl;
    }

    return idx;
}
