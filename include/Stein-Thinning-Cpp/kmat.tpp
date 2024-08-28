#include <armadillo>
#include "utils.h"
#include "kernel.h"

template <typename KernelFunction>
arma::mat stein_thinning::kmat(arma::mat &x, arma::mat &sx, arma::vec &x_map, arma::mat &linv, const KernelFunction kernel, const int s, const float beta)
{
    int n = x.n_rows;
    arma::mat K(n, n, arma::fill::zeros);

    for (int i = 0; i < n; i++)
    {
        for (int j = i; j < n; j++)
        {
            K(i, j) = kernel(x.row(i).t(), x.row(j).t(), sx.row(i).t(), sx.row(j).t(), x_map, linv, s, beta);
        }
    }
    stein_thinning::utils::mirror_lower(K);
    return K;
}

template <typename KernelFunction>
arma::mat stein_thinning::kmat(arma::mat &x, arma::mat &sx, arma::mat &linv, const KernelFunction kernel, const float beta)
{
    int n = x.n_rows;
    arma::mat K(n, n, arma::fill::zeros);

    for (int i = 0; i < n; i++)
    {
        for (int j = i; j < n; j++)
        {
            K(i, j) = kernel(x.row(i).t(), x.row(j).t(), sx.row(i).t(), sx.row(j).t(), linv, beta);
        }
    }
    stein_thinning::utils::mirror_lower(K);
    return K;
}

template <typename KernelFunction>
arma::mat stein_thinning::kmat(arma::mat &x, arma::mat &sx, arma::vec &x_map, const KernelFunction kernel)
{
    int n = x.n_rows;
    arma::mat K(n, n, arma::fill::zeros);

    for (int i = 0; i < n; i++)
    {
        for (int j = i; j < n; j++)
        {
            K(i, j) = kernel(x.row(i).t(), x.row(j).t(), sx.row(i).t(), sx.row(j).t(), x_map);
        }
    }
    stein_thinning::utils::mirror_lower(K);
    return K;
}

template <typename KernelFunction>
arma::mat stein_thinning::kmat(arma::mat &x, arma::mat &sx, const KernelFunction kernel)
{
    int n = x.n_rows;
    arma::mat K(n, n, arma::fill::zeros);

    for (int i = 0; i < n; i++)
    {
        for (int j = i; j < n; j++)
        {
            K(i, j) = kernel(x.row(i).t(), x.row(j).t(), sx.row(i).t(), sx.row(j).t());
        }
    }
    stein_thinning::utils::mirror_lower(K);
    return K;
}
