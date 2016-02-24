#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "stats/Distributions.h"

TEST(DISTRIBUTIONS_TEST, GAUSSIAN_PDF_FLOAT_TEST_CPU) {
    af::setBackend(AF_BACKEND_CPU);
    
    // normal bell curve
    ASSERT_FLOAT_EQ(juml::gaussian_pdf(-0.5f, 0.0f, 1.0f), 0.35206532676429952f);
    ASSERT_FLOAT_EQ(juml::gaussian_pdf(+0.0f, 0.0f, 1.0f), 0.39894228040143270f);
    ASSERT_FLOAT_EQ(juml::gaussian_pdf(+0.5f, 0.0f, 1.0f), 0.35206532676429952f);
    
    // different location
    ASSERT_FLOAT_EQ(juml::gaussian_pdf(+0.0f, +1.0f, 1.0f), 0.24197072451914337f);
    ASSERT_FLOAT_EQ(juml::gaussian_pdf(+0.5f, -1.0f, 1.0f), 0.12951759566589174f);
    
    // different standard deviation
    ASSERT_FLOAT_EQ(juml::gaussian_pdf(+0.5f, +1.0f, 0.5f), 0.48394144903828673f);
    ASSERT_FLOAT_EQ(juml::gaussian_pdf(+0.5f, +1.0f, 2.0f), 0.19333405840142465f);
}

TEST(DISTRIBUTIONS_TEST, GAUSSIAN_PDF_DOUBLE_TEST_CPU) {
    af::setBackend(AF_BACKEND_CPU);
    
    // normal bell curve
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf(-0.5, 0.0, 1.0), 0.35206532676429952);
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf(+0.0, 0.0, 1.0), 0.3989422804014327);
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf(+0.5, 0.0, 1.0), 0.35206532676429952);
    
    // different location
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf(+0.0, +1.0, 1.0), 0.24197072451914337);
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf(+0.5, -1.0, 1.0), 0.12951759566589174);
    
    // different standard deviation
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf(+0.5, +1.0, 0.5), 0.48394144903828673);
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf(+0.5, +1.0, 2.0), 0.19333405840142465);
}

TEST(DISTRIBUTIONS_TEST, GAUSSIAN_PDF_NAN_TEST_CPU) {
    af::setBackend(AF_BACKEND_CPU);
    const auto float_nan  = std::numeric_limits<float>::quiet_NaN();
    const auto double_nan = std::numeric_limits<double>::quiet_NaN();

    // passing NaN values
    ASSERT_TRUE(std::isnan(juml::gaussian_pdf(float_nan,  0.0f, 1.0f)));
    ASSERT_TRUE(std::isnan(juml::gaussian_pdf(double_nan, 0.0,  1.0)));
    
    // zero standard deviation
    ASSERT_TRUE(std::isnan(juml::gaussian_pdf(0.5f, 0.0f, 0.0f)));
    ASSERT_TRUE(std::isnan(juml::gaussian_pdf(0.5,  0.0,  0.0)));
}

TEST(DISTRIBUTIONS_TEST, GAUSSIAN_PDF_VECTOR_TEST_CPU) {
    af::setBackend(AF_BACKEND_CPU);
    
    const float val[] = {-0.5f, +0.0f, +0.5f};
    const float loc[] = {+0.5f, +0.0f, +-.5f};
    const float std[] = {+1.0f, +1.0f, +1.0f};
    const float exp[] = {0.24197072451914337f, 0.3989422804014327f, 0.24197072451914337f};
    
    af::array values    = af::array(3, val);
    af::array locations = af::array(3, loc);
    af::array stddevs   = af::array(3, std);
    af::array expected  = af::array(3, exp);
    af::array probs     = juml::gaussian_pdf(values, locations, stddevs);
    
    ASSERT_EQ(values.elements(), probs.elements());
    for (size_t i = 0; i < probs.elements(); ++i) {
        ASSERT_FLOAT_EQ(probs.host<float>()[i], expected.device<float>()[i]);
    }
}

#ifdef JUML_OPENCL
TEST(DISTRIBUTIONS_TEST, GAUSSIAN_PDF_FLOAT_TEST_OPENCL) {
    af::setBackend(AF_BACKEND_OPENCL);
    
    // normal bell curve
    ASSERT_FLOAT_EQ(juml::gaussian_pdf(-0.5f, 0.0f, 1.0f), 0.35206532676429952f);
    ASSERT_FLOAT_EQ(juml::gaussian_pdf(+0.0f, 0.0f, 1.0f), 0.39894228040143270f);
    ASSERT_FLOAT_EQ(juml::gaussian_pdf(+0.5f, 0.0f, 1.0f), 0.35206532676429952f);
    
    // different location
    ASSERT_FLOAT_EQ(juml::gaussian_pdf(+0.0f, +1.0f, 1.0f), 0.24197072451914337f);
    ASSERT_FLOAT_EQ(juml::gaussian_pdf(+0.5f, -1.0f, 1.0f), 0.12951759566589174f);
    
    // different standard deviation
    ASSERT_FLOAT_EQ(juml::gaussian_pdf(+0.5f, +1.0f, 0.5f), 0.48394144903828673f);
    ASSERT_FLOAT_EQ(juml::gaussian_pdf(+0.5f, +1.0f, 2.0f), 0.19333405840142465f);
}

TEST(DISTRIBUTIONS_TEST, GAUSSIAN_PDF_DOUBLE_TEST_OPENCL) {
    af::setBackend(AF_BACKEND_OPENCL);
    if (!af::isDoubleAvailable(af::getDevice())) {return;}
    
    // normal bell curve
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf(-0.5, 0.0, 1.0), 0.35206532676429952);
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf(+0.0, 0.0, 1.0), 0.3989422804014327);
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf(+0.5, 0.0, 1.0), 0.35206532676429952);
    
    // different location
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf(+0.0, +1.0, 1.0), 0.24197072451914337);
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf(+0.5, -1.0, 1.0), 0.12951759566589174);
    
    // different standard deviation
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf(+0.5, +1.0, 0.5), 0.48394144903828673);
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf(+0.5, +1.0, 2.0), 0.19333405840142465);
}

TEST(DISTRIBUTIONS_TEST, GAUSSIAN_PDF_NAN_TEST_OPENCL) {
    af::setBackend(AF_BACKEND_OPENCL);
    const auto float_nan  = std::numeric_limits<float>::quiet_NaN();
    const auto double_nan = std::numeric_limits<double>::quiet_NaN();

    // passing NaN values
    ASSERT_TRUE(std::isnan(juml::gaussian_pdf(float_nan,  0.0f, 1.0f)));
    if (af::isDoubleAvailable(af::getDevice())) {
        ASSERT_TRUE(std::isnan(juml::gaussian_pdf(double_nan, 0.0,  1.0)));
    }
    
    // zero standard deviation
    ASSERT_TRUE(std::isnan(juml::gaussian_pdf(0.5f, 0.0f, 0.0f)));
    if (af::isDoubleAvailable(af::getDevice())) {
        ASSERT_TRUE(std::isnan(juml::gaussian_pdf(0.5,  0.0,  0.0)));
    }
}

TEST(DISTRIBUTIONS_TEST, GAUSSIAN_PDF_VECTOR_TEST_OPENCL) {
    af::setBackend(AF_BACKEND_OPENCL);
    
    const float val[] = {-0.5f, +0.0f, +0.5f};
    const float loc[] = {+0.5f, +0.0f, +-.5f};
    const float std[] = {+1.0f, +1.0f, +1.0f};
    const float exp[] = {0.24197072451914337f, 0.3989422804014327f, 0.24197072451914337f};
    
    af::array values    = af::array(3, val);
    af::array locations = af::array(3, loc);
    af::array stddevs   = af::array(3, std);
    af::array expected  = af::array(3, exp);
    af::array probs     = juml::gaussian_pdf(values, locations, stddevs);
    
    ASSERT_EQ(values.elements(), probs.elements());
    for (size_t i = 0; i < probs.elements(); ++i) {
        ASSERT_FLOAT_EQ(probs.host<float>()[i], expected.host<float>()[i]);
    }
}
#endif

#ifdef JUML_CUDA
TEST(DISTRIBUTIONS_TEST, GAUSSIAN_PDF_FLOAT_TEST_CUDA) {
    af::setBackend(AF_BACKEND_CUDA);
    
    // normal bell curve
    ASSERT_FLOAT_EQ(juml::gaussian_pdf(-0.5f, 0.0f, 1.0f), 0.35206532676429952f);
    ASSERT_FLOAT_EQ(juml::gaussian_pdf(+0.0f, 0.0f, 1.0f), 0.39894228040143270f);
    ASSERT_FLOAT_EQ(juml::gaussian_pdf(+0.5f, 0.0f, 1.0f), 0.35206532676429952f);
    
    // different location
    ASSERT_FLOAT_EQ(juml::gaussian_pdf(+0.0f, +1.0f, 1.0f), 0.24197072451914337f);
    ASSERT_FLOAT_EQ(juml::gaussian_pdf(+0.5f, -1.0f, 1.0f), 0.12951759566589174f);
    
    // different standard deviation
    ASSERT_FLOAT_EQ(juml::gaussian_pdf(+0.5f, +1.0f, 0.5f), 0.48394144903828673f);
    ASSERT_FLOAT_EQ(juml::gaussian_pdf(+0.5f, +1.0f, 2.0f), 0.19333405840142465f);
}

TEST(DISTRIBUTIONS_TEST, GAUSSIAN_PDF_DOUBLE_TEST_CUDA) {
    af::setBackend(AF_BACKEND_CUDA);
    if (!af::isDoubleAvailable(af::getDevice())) {return;}
    
    // normal bell curve
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf(-0.5, 0.0, 1.0), 0.35206532676429952);
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf(+0.0, 0.0, 1.0), 0.3989422804014327);
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf(+0.5, 0.0, 1.0), 0.35206532676429952);
    
    // different location
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf(+0.0, +1.0, 1.0), 0.24197072451914337);
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf(+0.5, -1.0, 1.0), 0.12951759566589174);
    
    // different standard deviation
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf(+0.5, +1.0, 0.5), 0.48394144903828673);
    ASSERT_DOUBLE_EQ(juml::gaussian_pdf(+0.5, +1.0, 2.0), 0.19333405840142465);
}

TEST(DISTRIBUTIONS_TEST, GAUSSIAN_PDF_NAN_TEST_CUDA) {
    af::setBackend(AF_BACKEND_CUDA);
    const auto float_nan  = std::numeric_limits<float>::quiet_NaN();
    const auto double_nan = std::numeric_limits<double>::quiet_NaN();

    // passing NaN values
    ASSERT_TRUE(std::isnan(juml::gaussian_pdf(float_nan,  0.0f, 1.0f)));
    if (af::isDoubleAvailable(af::getDevice())) {
        ASSERT_TRUE(std::isnan(juml::gaussian_pdf(double_nan, 0.0,  1.0)));
    }
    
    // zero standard deviation
    ASSERT_TRUE(std::isnan(juml::gaussian_pdf(0.5f, 0.0f, 0.0f)));
    if (af::isDoubleAvailable(af::getDevice())) {
        ASSERT_TRUE(std::isnan(juml::gaussian_pdf(0.5,  0.0,  0.0)));
    }
}

TEST(DISTRIBUTIONS_TEST, GAUSSIAN_PDF_VECTOR_TEST_CUDA) {
    af::setBackend(AF_BACKEND_CUDA);
    
    const float val[] = {-0.5f, +0.0f, +0.5f};
    const float loc[] = {+0.5f, +0.0f, +-.5f};
    const float std[] = {+1.0f, +1.0f, +1.0f};
    const float exp[] = {0.24197072451914337f, 0.3989422804014327f, 0.24197072451914337f};
    
    af::array values    = af::array(3, val);
    af::array locations = af::array(3, loc);
    af::array stddevs   = af::array(3, std);
    af::array expected  = af::array(3, exp);
    af::array probs     = juml::gaussian_pdf(values, locations, stddevs);
    
    ASSERT_EQ(values.elements(), probs.elements());
    for (size_t i = 0; i < probs.elements(); ++i) {
        ASSERT_FLOAT_EQ(probs.host<float>()[i], expected.host<float>()[i]);
    }
}
#endif

int main(int argc, char** argv) {
    int result = -1;
    
    ::testing::InitGoogleTest(&argc, argv);
    try {
        result = RUN_ALL_TESTS();
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
    }
    
    return result;
}

