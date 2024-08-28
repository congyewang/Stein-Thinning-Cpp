#include <armadillo>
#include "utils.h"

void stein_thinning::utils::mirror_lower(arma::mat &a)
{
    int n = a.n_rows;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < i; j++)
        {
            a(i, j) = a(j, i);
        }
    }
}
