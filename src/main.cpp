#include <armadillo>
#include "kernel/kernel.h"
#include "kernel/thinning.h"

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <path_to_smp.csv> <path_to_scr.csv>" << std::endl;
        return 1;
    }

    arma::mat smp;
    arma::mat scr;

    bool status_smp = smp.load(argv[1]);
    if (!status_smp)
    {
        std::cerr << "Failed to load " << argv[1] << std::endl;
        return 1;
    }

    bool status_scr = scr.load(argv[2]);
    if (!status_scr)
    {
        std::cerr << "Failed to load " << argv[2] << std::endl;
        return 1;
    }

    arma::uvec idx = thin(smp, scr, 3);
    idx.print("idx: ");

    return 0;
}
