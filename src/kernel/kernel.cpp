//
// Created by congye on 5/11/23.
//

#include <armadillo>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdexcept>

float vfk0_centkgm(const arma::vec &x, const arma::vec &y, const arma::vec &sx, const arma::vec &sy, const arma::vec &x_map,
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

float vfk0_imq(const arma::vec &x, const arma::vec &y, const arma::vec &sx, const arma::vec &sy, const arma::mat &linv,
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

double med2(const arma::mat& smp, int sz, int m) {
    arma::mat sub;
    if (sz > m) {
        // Subsampling
        sub.set_size(m, smp.n_cols);
        for(int i = 0; i < m; ++i) {
            sub.row(i) = smp.row(i * sz / m);
        }
    } else {
        sub = smp;
    }

    // Compute pairwise distances and find median
    std::vector<double> distances;
    for(int i = 0; i < sub.n_rows; ++i) {
        for(int j = i + 1; j < sub.n_rows; ++j) {
            distances.push_back(arma::norm(sub.row(i) - sub.row(j), 2));
        }
    }

    std::nth_element(distances.begin(), distances.begin() + distances.size() / 2, distances.end());
    return std::pow(distances[distances.size() / 2], 2);
}

arma::mat make_precon(const arma::mat& smp, const arma::mat& scr, const std::string& pre = "id") {
    // Sample size and dimension
    int sz = smp.n_rows;
    int dm = smp.n_cols;

    // Select preconditioner
    arma::mat linv;
    int m = 1000;
    if(pre == "id") {
        linv = arma::eye(dm, dm);
    } else if(pre == "med" || pre == "sclmed") {
        double m2 = med2(smp, sz, m);
        if(m2 == 0)
            throw std::runtime_error("Too few unique samples in smp.");
        if(pre == "med")
            linv = arma::inv(m2 * arma::eye(dm, dm));
        else if(pre == "sclmed")
            linv = arma::inv((m2 / std::log(std::min(m, sz))) * arma::eye(dm, dm));
    } else if(pre == "smpcov") {
        arma::mat c = arma::cov(smp);  // Compute covariance matrix
        arma::vec eigval = arma::eig_sym(c);
        if(eigval.min() <= 0)
            throw std::runtime_error("Too few unique samples in smp.");
        linv = arma::inv(c);
    } else {
        try {
            double preVal = std::stod(pre);
            linv = arma::inv(preVal * arma::eye(dm, dm));
        } catch(const std::invalid_argument& e) {
            throw std::invalid_argument("Incorrect preconditioner string.");
        }
    }
    return linv;
}

// Function Object for vfk0_imq
class Vfk0Imq {
public:
    Vfk0Imq(const arma::vec &x, const arma::vec &y, const arma::vec &sx, const arma::vec &sy, const arma::mat &linv)
            : x_(x), y_(y), sx_(sx), sy_(sy), linv_(linv) {}

    float operator()(float beta = 0.5) const {
        return vfk0_imq(x_, y_, sx_, sy_, linv_, beta);
    }

private:
    arma::vec x_;
    arma::vec y_;
    arma::vec sx_;
    arma::vec sy_;
    arma::mat linv_;
};

// Function Object for vfk0_centkgm
class Vfk0Centkgm {
public:
    Vfk0Centkgm(const arma::vec &x, const arma::vec &y, const arma::vec &sx, const arma::vec &sy, const arma::vec &x_map,
                const arma::mat &linv, float s = 3.0)
            : x_(x), y_(y), sx_(sx), sy_(sy), x_map_(x_map), linv_(linv), s_(s) {}

    float operator()(float beta = 0.5) const {
        return vfk0_centkgm(x_, y_, sx_, sy_, x_map_, linv_, s_, beta);
    }

private:
    arma::vec x_;
    arma::vec y_;
    arma::vec sx_;
    arma::vec sy_;
    arma::vec x_map_;
    arma::mat linv_;
    float s_;
};

Vfk0Imq make_imq(const arma::mat& smp, const arma::mat& scr, const std::string& pre = "id") {
    arma::mat linv = make_precon(smp, scr, pre);
    // Here you can replace x, y, sx, sy with your own data
    return Vfk0Imq(x, y, sx, sy, linv);
}

Vfk0Centkgm make_centkgm(const arma::mat& smp, const arma::mat& scr, const arma::vec& x_map, const std::string& pre = "id", float s = 3.0) {
    arma::mat linv = make_precon(smp, scr, pre);
    // Here you can replace x, y, sx, sy with your own data
    return Vfk0Centkgm(x, y, sx, sy, x_map, linv, s);
}
