#ifndef MAIN_STEIN_H
#define MAIN_STEIN_H

#include <armadillo>

arma::vec vfps_centkgm(const arma::mat &x_new, const arma::mat &sx_new, const arma::mat &x, const arma::mat &sx, const arma::vec &x_map, const int i, const std::function<arma::vec(const arma::mat &x, const arma::mat &y, const arma::mat &sx, const arma::mat &sy, const arma::vec &x_map)> vfk0);

arma::vec vfps_imq(const arma::mat &x_new, const arma::mat &sx_new, const arma::mat &x, const arma::mat &sx, const int i, const std::function<arma::vec(const arma::mat &x, const arma::mat &y, const arma::mat &sx, const arma::mat &sy)> vfk0);

#endif // MAIN_STEIN_H