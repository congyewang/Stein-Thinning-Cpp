#include <armadillo>
#include <functional>

template <typename VFK0Func>
arma::vec vfps_generic(const arma::mat &x_new, const arma::mat &sx_new,
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

arma::vec vfps_centkgm(const arma::mat &x_new, const arma::mat &sx_new,
                       const arma::mat &x, const arma::mat &sx,
                       const arma::vec &x_map, const int i,
                       const std::function<arma::vec(const arma::mat &x, const arma::mat &y,
                                                     const arma::mat &sx, const arma::mat &sy,
                                                     const arma::vec &x_map)>
                           vfk0)
{
    return vfps_generic(x_new, sx_new, x, sx, x_map, i, vfk0);
}

arma::vec vfps_imq(const arma::mat &x_new, const arma::mat &sx_new,
                   const arma::mat &x, const arma::mat &sx, const int i,
                   const std::function<arma::vec(const arma::mat &x, const arma::mat &y,
                                                 const arma::mat &sx, const arma::mat &sy)>
                       vfk0)
{
    return vfps_generic(x_new, sx_new, x, sx, {}, i,
                        [&vfk0](const arma::mat &xa, const arma::mat &xb,
                                const arma::mat &sxa, const arma::mat &sxb, const arma::vec &)
                        {
                            return vfk0(xa, xb, sxa, sxb);
                        });
}
