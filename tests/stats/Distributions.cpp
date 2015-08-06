#include <gtest/gtest.h>

#include "stats/Distributions.h"

TEST (DISTRIBUTIONS_TEST, GAUSSIAN_PDF_TEST) {
    ASSERT_EQ(juml::gaussian_pdf<float>(0.5f, 0.0f, 1.0f), 0.35206532676429952f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
