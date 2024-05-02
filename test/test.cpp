#include <armadillo>
#include <cassert>
#include "kernel/kernel.h"
#include "kernel/thinning.h"

int main(void)
{

    int m = 40;
    arma::mat smp;
    arma::mat scr;

    bool status_smp = smp.load("../../demo/sample_chains/gmm/smp.csv");
    if (!status_smp)
    {
        std::cerr << "Failed to load smp.csv" << std::endl;
        return 1;
    }

    bool status_scr = scr.load("../../demo/sample_chains/gmm/scr.csv");
    if (!status_scr)
    {
        std::cerr << "Failed to load scr.csv" << std::endl;
        return 1;
    }

    arma::uvec idx = thin(smp, scr, m);

    arma::uvec gold_standard = {
         68, 322, 268, 234, 161, 292, 229, 275,
        259, 131, 400, 486, 207, 120, 443, 430,
        376, 411,  98, 293, 111, 372, 285, 427,
        406, 246, 148, 260, 296, 208,  79, 430,
        369, 363, 462, 393, 321, 460, 373, 114
        };

    if (arma::all(idx == gold_standard)) {
        std::cout << "Passed Test." << std::endl;
    } else {
        throw std::runtime_error("Thinning Index and Gold Standard are Not Identical!");
    }

    return 0;
}
