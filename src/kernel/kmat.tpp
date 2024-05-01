//
// Created by congye on 7/13/23.
//

#include <armadillo>

void mirror_lower(arma::mat &a)
{
    int n = a.n_rows;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < i; j++)
        {
            a(i, j) = a(j, i);
        }
    }
}

template <typename KernelFunction>
arma::mat kmat(arma::mat &x, arma::mat &sx, arma::vec &x_map, arma::mat &linv, KernelFunction kernel, float s = 3.0, float beta = 0.5)
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
    mirror_lower(K);
    return K;
}

template <typename KernelFunction>
arma::mat kmat(arma::mat &x, arma::mat &sx, arma::mat &linv, KernelFunction kernel, float beta = 0.5)
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
    mirror_lower(K);
    return K;
}
