#ifndef MAIN_THINNING_H
#define MAIN_THINNING_H

#include <armadillo>

namespace stein_thinning
{

    arma::uvec thin(const arma::mat &smp, const arma::mat &scr, const int m, const std::string &pre = "id", const bool stnd = true, const bool verb = false);

}

#endif // MAIN_THINNING_H
