#include <gtest/gtest.h>
#include <armadillo>
#include "kernel/kernel.h"
#include "kernel/thinning.h"

// Defining Test Cases
TEST(ThinningTest, ThinningCorrectness)
{

    int m = 40;
    arma::mat smp;
    arma::mat scr;

    // Load smp.csv
    bool status_smp = smp.load("../../demo/sample_chains/gmm/smp.csv");
    ASSERT_TRUE(status_smp) << "Failed to load smp.csv";

    // Load scr.csv
    bool status_scr = scr.load("../../demo/sample_chains/gmm/scr.csv");
    ASSERT_TRUE(status_scr) << "Failed to load scr.csv";

    // Thinning
    arma::uvec idx = thin(smp, scr, m);

    // Define gold_standard
    arma::uvec gold_standard = {
        68, 322, 268, 234, 161, 292, 229, 275,
        259, 131, 400, 486, 207, 120, 443, 430,
        376, 411, 98, 293, 111, 372, 285, 427,
        406, 246, 148, 260, 296, 208, 79, 430,
        369, 363, 462, 393, 321, 460, 373, 114};

    // Valid if idx and gold_standard are identical
    EXPECT_TRUE(arma::all(idx == gold_standard)) << "Thinning Index and Gold Standard are Not Identical!";
}

// Main Function
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
