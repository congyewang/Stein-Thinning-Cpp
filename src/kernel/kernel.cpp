//
// Created by congye on 5/11/23.
//

#include <armadillo>

float kp_kgm(const arma::vec &x, const arma::vec &y, const arma::vec &sx, const arma::vec &sy, const arma::vec &x_map,
             const arma::mat &linv, float s = 3.0, float beta = 0.5) {
    arma::vec kappa, dxkappa, dykappa, dxdykappa, c, dxc, dyc, dxdyc, kp;
    float res;

    kappa = arma::pow(1 + (x - y).t() * linv * (x - y), -beta) +
            (1 + (x - x_map).t() * linv * (y - x_map)) / (arma::pow(1 + (x - x_map).t() * linv * (x - x_map), s / 2) *
                                                          arma::pow(1 + (y - x_map).t() * linv * (y - x_map), s / 2));

    dxkappa = -2 * beta * arma::as_scalar(arma::pow(1 + (x - y).t() * linv * (x - y), -beta - 1)) * linv * (x - y) +
              (linv * (y - x_map) -
               s * (1 + arma::as_scalar((x - x_map).t() * linv * (y - x_map))) * linv * (x - x_map) *
               arma::as_scalar(arma::pow(1 + (x - x_map).t() * linv * (x - x_map), -1))) /
              arma::as_scalar(
                      arma::pow(1 + (x - x_map).t() * linv * (x - x_map), s / 2) *
                      arma::pow(1 + (y - x_map).t() * linv * (y - x_map), s / 2));

    dykappa = 2 * beta * arma::as_scalar(arma::pow(1 + (x - y).t() * linv * (x - y), -beta - 1)) * linv * (x - y) +
              (linv * (x - x_map) -
               s * (1 + arma::as_scalar((x - x_map).t() * linv * (y - x_map))) * linv * (y - x_map) *
               arma::as_scalar(arma::pow(1 + (y - x_map).t() * linv * (y - x_map), -1))) /
              arma::as_scalar(
                      arma::pow(1 + (x - x_map).t() * linv * (x - x_map), s / 2) *
                      arma::pow(1 + (y - x_map).t() * linv * (y - x_map), s / 2));

    dxdykappa =
            -4 * beta * (beta + 1) * arma::pow(1 + (x - y).t() * linv * (x - y), -beta - 2) * (x - y).t() * linv *
            linv *
            (x - y) +
            2 * beta * trace(linv) * arma::pow(1 + (x - y).t() * linv * (x - y), -beta - 1) +
            (trace(linv) -
             s * arma::pow(1 + (x - x_map).t() * linv * (x - x_map), -1) * (x - x_map).t() * linv * linv * (x - x_map) -
             s * arma::pow(1 + (y - x_map).t() * linv * (y - x_map), -1) * (y - x_map).t() * linv * linv * (y - x_map) +
             pow(s, 2) * (1 + (x - x_map).t() * linv * (y - x_map)) *
             arma::pow(1 + (x - x_map).t() * linv * (x - x_map), -1) *
             arma::pow(1 + (y - x_map).t() * linv * (y - x_map), -1) * ((x - x_map).t() * linv * linv * (y - x_map))) /
            (arma::pow(1 + (x - x_map).t() * linv * (x - x_map), s / 2) *
             arma::pow(1 + (y - x_map).t() * linv * (y - x_map), s / 2));

    c = arma::pow(1 + (x - x_map).t() * linv * (x - x_map), (s - 1) / 2) *
        arma::pow(1 + (y - x_map).t() * linv * (y - x_map), (s - 1) / 2) * kappa;

    dxc = arma::as_scalar(arma::pow(1 + (x - x_map).t() * linv * (x - x_map), (s - 1) / 2)) *
          arma::as_scalar(arma::pow(1 + (y - x_map).t() * linv * (y - x_map), (s - 1) / 2)) *
          (((s - 1) * arma::as_scalar(kappa) * linv * (x - x_map)) /
           arma::as_scalar(1 + (x - x_map).t() * linv * (x - x_map)) + dxkappa);

    dyc = arma::as_scalar(arma::pow(1 + (x - x_map).t() * linv * (x - x_map), (s - 1) / 2)) *
          arma::as_scalar(arma::pow(1 + (y - x_map).t() * linv * (y - x_map), (s - 1) / 2)) *
          (((s - 1) * arma::as_scalar(kappa) * linv * (y - x_map)) /
           arma::as_scalar(1 + (y - x_map).t() * linv * (y - x_map)) + dykappa);

    dxdyc = arma::pow(1 + (x - x_map).t() * linv * (x - x_map), (s - 1) / 2) *
            arma::pow(1 + (y - x_map).t() * linv * (y - x_map), (s - 1) / 2) *
            (pow(s - 1, 2) * kappa * (x - x_map).t() * linv * linv * (y - x_map) /
             ((1 + (x - x_map).t() * linv * (x - x_map)) * (1 + (y - x_map).t() * linv * (y - x_map))) +
             (s - 1) * (y - x_map).t() * linv * dxkappa / (1 + (y - x_map).t() * linv * (y - x_map)) +
             (s - 1) * (x - x_map).t() * linv * dykappa / (1 + (x - x_map).t() * linv * (x - x_map)) + dxdykappa);

    kp = dxdyc + dxc.t() * sy + dyc.t() * sx + c * sx.t() * sy;
    res = kp[0];

    return res;
}

float kp_imq(const arma::vec &x, const arma::vec &y, const arma::vec &sx, const arma::vec &sy, const arma::mat &linv,
             float beta = 0.5) {
    arma::vec res1, res2, res3, res4, res5, res6, kp;
    res1 = 4 * beta * (beta + 1) * ((x - y).t() * arma::powmat(linv, 2) * (x - y));
    res2 = arma::pow(1 + (x - y).t() * linv * (x - y), beta + 2);
    res3 = arma::trace(linv) + (sx - sy).t() * linv * (x - y);
    res4 = arma::pow(1 + (x - y).t() * linv * (x - y), 1 + beta);
    res5 = sx.t() * sy;
    res6 = arma::pow(1 + (x - y).t() * linv * (x - y), beta);
    kp = -res1 / res2 + 2 * beta * (res3 / res4) + res5 / res6;

    float res = kp[0];

    return (res);
}
