#include <armadillo>
#include <functional>

template <typename VFK0Func>
arma::vec vfps(const arma::mat &x_new, const arma::mat &sx_new,
               const arma::mat &x, const arma::mat &sx,
               const arma::vec &x_map, const int i, VFK0Func vfk0)
{
    arma::vec k0aa = vfk0(x_new, x_new, sx_new, sx_new, x_map);
    int n_new = x_new.n_rows;

    arma::mat a;
    arma::mat b;
    arma::mat sa;
    arma::mat sb;
    arma::mat k0ab;

    arma::vec res;

    if (i > 0)
    {
        a = arma::repmat(x_new, i, 1);
        b = arma::repelem(x.rows(0, i - 1), n_new, 1);
        sa = arma::repmat(sx_new, i, 1);
        sb = arma::repelem(sx.rows(0, i - 1), n_new, 1);
        k0ab = arma::reshape(vfk0(a, b, sa, sb, x_map), n_new, i).t();

        res = arma::sum(k0ab, 0).t() * 2 + k0aa;
    }
    else
    {
        res = k0aa;
    }
    return res;
}

template <typename VFK0Func>
arma::vec vfps(const arma::mat &x_new, const arma::mat &sx_new,
               const arma::mat &x, const arma::mat &sx,
               const int i, VFK0Func vfk0)
{
    arma::vec k0aa = vfk0(x_new, x_new, sx_new, sx_new);
    int n_new = x_new.n_rows;

    arma::mat a;
    arma::mat b;
    arma::mat sa;
    arma::mat sb;
    arma::mat k0ab;

    arma::vec res;

    if (i > 0)
    {
        a = arma::repmat(x_new, i, 1);
        b = arma::repelem(x.rows(0, i - 1), n_new, 1);
        sa = arma::repmat(sx_new, i, 1);
        sb = arma::repelem(sx.rows(0, i - 1), n_new, 1);
        k0ab = arma::reshape(vfk0(a, b, sa, sb), n_new, i).t();

        res = arma::sum(k0ab, 0).t() * 2 + k0aa;
    }
    else
    {
        res = k0aa;
    }
    return res;
}
