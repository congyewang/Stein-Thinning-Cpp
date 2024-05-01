//
// Created by congye on 5/11/23.
//

#ifndef MAIN_KERNEL_H
#define MAIN_KERNEL_H

#include <armadillo>
#include "kmat.tpp"

float vfk0_centkgm(const arma::vec &x, const arma::vec &y, const arma::vec &sx, const arma::vec &sy, const arma::vec &x_map,
                   const arma::mat &linv, const float s = 3.0, const float beta = 0.5);

float vfk0_imq(const arma::vec &x, const arma::vec &y, const arma::vec &sx, const arma::vec &sy, const arma::mat &linv,
               const float beta = 0.5);

double med2(const arma::mat &smp, int sz, int m);

arma::mat make_precon(const arma::mat &smp, const arma::mat &scr, const std::string &pre);

template <typename KernelFunction>
arma::mat kmat(arma::mat &x, arma::mat &sx, arma::vec &x_map, arma::mat &linv, KernelFunction kernel, float s, float beta);

template <typename KernelFunction>
arma::mat kmat(arma::mat &x, arma::mat &sx, arma::mat &linv, KernelFunction kernel, float beta);

std::function<float(const arma::vec &, const arma::vec &, const arma::vec &, const arma::vec &, const float)> make_imq(
    const arma::mat &smp, const arma::mat &scr, const std::string &pre = "id");

std::function<float(const arma::vec &, const arma::vec &, const arma::vec &, const arma::vec &, const arma::vec &, const float)> make_centkgm(
    const arma::mat &smp, const arma::mat &scr, const std::string &pre = "id");

#endif // MAIN_KERNEL_H
