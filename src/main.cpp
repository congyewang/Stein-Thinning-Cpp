#include <armadillo>
#include "kernel.h"
#include "thinning.h"

int main(int argc, char **argv)
{
    if (argc < 4 || argc > 5)
    {
        std::cerr << "Usage: " << argv[0] << "<int number_of_points> <str /path/to/smp.csv> <str /path/to/scr.csv> [str /path/to/output.csv]" << std::endl;
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

    std::string output_path = (argc == 5) ? argv[4] : "./output.csv";
    idx.save(arma::csv_name(output_path, arma::csv_opts::no_header));
    std::cout << "Results saved to: " << output_path << std::endl;

    return 0;
}
