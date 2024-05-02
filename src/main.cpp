#include <armadillo>
#include "kernel/kernel.h"
#include "kernel/thinning.h"

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << "<int number_of_points> <str path_to_smp.csv> <str path_to_scr.csv>" << std::endl;
        return 1;
    }
    int m;
    arma::mat smp;
    arma::mat scr;

    m = std::atoi(argv[1]);

    bool status_smp = smp.load(argv[2]);
    if (!status_smp)
    {
        std::cerr << "Failed to load " << argv[1] << std::endl;
        return 1;
    }

    bool status_scr = scr.load(argv[3]);
    if (!status_scr)
    {
        std::cerr << "Failed to load " << argv[2] << std::endl;
        return 1;
    }

    arma::uvec idx = thin(smp, scr, m);
    idx.print("idx: ");

    return 0;
}
