//
// Created by congye on 5/11/23.
//

#ifndef MAIN_KERNEL_H
#define MAIN_KERNEL_H

#include <armadillo>
#include "kmat.tpp"

float vfk0_centkgm(const arma::vec &x, const arma::vec &y, const arma::vec &sx, const arma::vec &sy, const arma::vec &x_map,
             const arma::mat &linv, float s = 3.0, float beta = 0.5);

float vfk0_imq(const arma::vec &x, const arma::vec &y, const arma::vec &sx, const arma::vec &sy, const arma::mat &linv,
             float beta = 0.5);

double med2(const arma::mat& smp, int sz, int m);

arma::mat make_precon(const arma::mat& smp, const arma::mat& scr, const std::string& pre);

template<typename KernelFunction>
arma::mat kmat(arma::mat &x, arma::mat &sx, arma::vec &x_map, arma::mat &linv, KernelFunction kernel, float s, float beta);

template<typename KernelFunction>
arma::mat kmat(arma::mat &x, arma::mat &sx, arma::mat &linv, KernelFunction kernel, float beta);

class Vfk0Imq {
public:
    Vfk0Imq(const arma::vec &x, const arma::vec &y, const arma::vec &sx, const arma::vec &sy, const arma::mat &linv)
            : x_(x), y_(y), sx_(sx), sy_(sy), linv_(linv) {}

    float operator()(float beta = 0.5) const;

private:
    arma::vec x_;
    arma::vec y_;
    arma::vec sx_;
    arma::vec sy_;
    arma::mat linv_;
};

class Vfk0Centkgm {
public:
    Vfk0Centkgm(const arma::vec &x, const arma::vec &y, const arma::vec &sx, const arma::vec &sy, const arma::vec &x_map,
                const arma::mat &linv, float s = 3.0)
            : x_(x), y_(y), sx_(sx), sy_(sy), x_map_(x_map), linv_(linv), s_(s) {}

    float operator()(float beta = 0.5) const;

private:
    arma::vec x_;
    arma::vec y_;
    arma::vec sx_;
    arma::vec sy_;
    arma::vec x_map_;
    arma::mat linv_;
    float s_;
};

Vfk0Imq make_imq(const arma::mat& smp, const arma::mat& scr, const std::string& pre = "id");

Vfk0Centkgm make_centkgm(const arma::mat& smp, const arma::mat& scr, const arma::vec& x_map, const std::string& pre = "id", float s = 3.0);

#endif //MAIN_KERNEL_H
