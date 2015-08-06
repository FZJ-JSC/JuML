#include <gtest/gtest.h>

#include <armadillo>
#include <cmath>
#include <limits>

#include "stats/Distributions.h"

TEST (DISTRIBUTIONS_TEST, GAUSSIAN_PDF_FLOAT_TEST) {
    // normal bell curve
    ASSERT_FLOAT_EQ(juml::gaussian_pdf<float>(-0.5f, 0.0f, 1.0f), 0.35206532676429952f);
    ASSERT_FLOAT_EQ(juml::gaussian_pdf<float>(+0.0f, 0.0f, 1.0f), 0.3989422804014327f);
    ASSERT_FLOAT_EQ(juml::gaussian_pdf<float>(+0.5f, 0.0f, 1.0f), 0.35206532676429952f);
    
    // different location
    ASSERT_FLOAT_EQ(juml::gaussian_pdf<float>(+0.0f, +1.0f, 1.0f), 0.24197072451914337f);
    ASSERT_FLOAT_EQ(juml::gaussian_pdf<float>(+0.5f, -1.0f, 1.0f), 0.12951759566589174f);
    
    // different standard deviation
    ASSERT_FLOAT_EQ(juml::gaussian_pdf<float>(+0.5f, +1.0f, 0.5f), 0.48394144903828673f);
    ASSERT_FLOAT_EQ(juml::gaussian_pdf<float>(+0.5f, +1.0f, 2.0f), 0.19333405840142465f);
}

TEST (DISTRIBUTIONS_TEST, GAUSSIAN_PDF_DOUBLE_TEST) {
    // normal bell curve
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf<double>(-0.5, 0.0, 1.0), 0.35206532676429952);
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf<double>(+0.0, 0.0, 1.0), 0.3989422804014327);
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf<double>(+0.5, 0.0, 1.0), 0.35206532676429952);
    
    // different location
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf<double>(+0.0, +1.0, 1.0), 0.24197072451914337);
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf<double>(+0.5, -1.0, 1.0), 0.12951759566589174);
    
    // different standard deviation
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf<double>(+0.5, +1.0, 0.5), 0.48394144903828673);
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf<double>(+0.5, +1.0, 2.0), 0.19333405840142465);
}

TEST (DISTRIBUTIONS_TEST, GAUSSIAN_PDF_NAN_TEST) {
    const auto float_nan  = std::numeric_limits<float>::quiet_NaN();
    const auto double_nan = std::numeric_limits<double>::quiet_NaN();

    // passing NaN values
    ASSERT_TRUE(std::isnan(juml::gaussian_pdf<float>(float_nan, 0.0f, 1.0f)));
    ASSERT_TRUE(std::isnan(juml::gaussian_pdf<double>(double_nan, 0.0, 1.0)));
    
    // zero standard deviation
    ASSERT_TRUE(std::isnan(juml::gaussian_pdf<float>(0.5f, 0.0f, 0.0f)));
    ASSERT_TRUE(std::isnan(juml::gaussian_pdf<double>(0.5, 0.0, 0.0)));
}

TEST (DISTRIBUTIONS_TEST, GAUSSIAN_PDF_VECTOR_TEST) {
    arma::Row<float> values    = {-0.5f, +0.0f, +0.5f};
    arma::Row<float> locations = {+0.5f, +0.0f, -0.5f};
    arma::Row<float> stddevs   = {+1.0f, +1.0f, +1.0f};
    arma::Row<float> probs     = juml::gaussian_pdf(values, locations, stddevs);
    arma::Row<float> expected  = {
        0.24197072451914337f,
        0.3989422804014327f,
        0.24197072451914337f
    };
    
    ASSERT_EQ(values.n_elem, probs.n_elem);
    for (size_t i = 0; i < values.n_elem; ++i) {
        ASSERT_FLOAT_EQ(probs[i], expected[i]);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

