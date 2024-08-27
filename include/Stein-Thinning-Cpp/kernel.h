#ifndef MAIN_KERNEL_H
#define MAIN_KERNEL_H

#include <armadillo>
#include "kmat.tpp"
#include "stein.tpp"

double stein_kernel_centkgm(const arma::vec &x, const arma::vec &y, const arma::vec &sx, const arma::vec &sy, const arma::vec &x_map,
                            const arma::mat &linv, const int s = 3.0, const float beta = 0.5);

double stein_kernel_imq(const arma::vec &x, const arma::vec &y, const arma::vec &sx, const arma::vec &sy, const arma::mat &linv,
                        const float beta = 0.5);

double med2(const arma::mat &smp, int sz, int m);

arma::mat make_precon(const arma::mat &smp, const arma::mat &scr, const std::string &pre);

template <typename KernelFunction>
arma::mat kmat(arma::mat &x, arma::mat &sx, arma::vec &x_map, arma::mat &linv, KernelFunction kernel, int s, float beta);

template <typename KernelFunction>
arma::mat kmat(arma::mat &x, arma::mat &sx, arma::mat &linv, KernelFunction kernel);

template <typename KernelFunction>
arma::mat kmat(arma::mat &x, arma::mat &sx, arma::vec &x_map, const KernelFunction kernel);

template <typename KernelFunction>
arma::mat kmat(arma::mat &x, arma::mat &sx, KernelFunction kernel);

std::function<float(const arma::vec &x, const arma::vec &y, const arma::vec &sx, const arma::vec &sy, const arma::vec &x_map)> make_centkgm(
    const arma::mat &smp, const arma::mat &scr, const std::string &pre = "id");

std::function<float(const arma::vec &x, const arma::vec &y, const arma::vec &sx, const arma::vec &sy)> make_imq(
    const arma::mat &smp, const arma::mat &scr, const std::string &pre = "id");

arma::vec vectorised_stein_kernel_centkgm(const arma::mat &x, const arma::mat &y, const arma::mat &sx, const arma::mat &sy, const arma::vec &x_map, const std::string &pre = "id");

arma::vec vectorised_stein_kernel_imq(const arma::mat &x, const arma::mat &y, const arma::mat &sx, const arma::mat &sy, const std::string &pre = "id");

arma::vec vfps(const arma::mat &x_new, const arma::mat &sx_new, const arma::mat &x, const arma::mat &sx, const int i, std::function<arma::vec(const arma::mat &x, const arma::mat &y, const arma::mat &sx, const arma::mat &sy)> vfk0);

template <typename VFK0Func>
arma::vec vfps(const arma::mat &x_new, const arma::mat &sx_new,
               const arma::mat &x, const arma::mat &sx,
               const arma::vec &x_map, const int i, VFK0Func vfk0);

template <typename VFK0Func>
arma::vec vfps(const arma::mat &x_new, const arma::mat &sx_new,
               const arma::mat &x, const arma::mat &sx,
               const int i, VFK0Func vfk0);

#endif // MAIN_KERNEL_H
