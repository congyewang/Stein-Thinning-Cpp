//
// Created by congye on 5/11/23.
//

#ifndef MAIN_KERNEL_H
#define MAIN_KERNEL_H

#include <armadillo>
#include "kmat.tpp"

float kp_kgm(const arma::vec &x, const arma::vec &y, const arma::vec &sx, const arma::vec &sy, const arma::vec &x_map,
             const arma::mat &linv, float s = 3.0, float beta = 0.5);

float kp_imq(const arma::vec &x, const arma::vec &y, const arma::vec &sx, const arma::vec &sy, const arma::mat &linv,
             float beta = 0.5);

template<typename KernelFunction>
arma::mat kmat(arma::mat &x, arma::mat &sx, arma::vec &x_map, arma::mat &linv, KernelFunction kernel, float s, float beta);

template<typename KernelFunction>
arma::mat kmat(arma::mat &x, arma::mat &sx, arma::mat &linv, KernelFunction kernel, float beta);

#endif //MAIN_KERNEL_H
