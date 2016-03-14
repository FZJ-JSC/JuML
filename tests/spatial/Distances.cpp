#include <arrayfire.h>
#include <gtest/gtest.h>
#include <stdexcept>

#include "spatial/Distances.h"

const float DESTINATIONS[6][3] = {{0.0f, 0.0f, 0.0f},
                                  {0.0f, 0.0f, 1.0f},
                                  {0.0f, 1.0f, 0.0f},
                                  {1.0f, 0.0f, 0.0f},
                                  {1.0f, 1.0f, 1.0f},
                                  {-1.f, -1.f, -1.f}};

const float ORIGINS[2][3] = {{0.0f, 0.0f, 0.0f},
                             {1.0f, 1.0f, 1.0f}};

const float EUCLIDEAN_DISTANCES[6][2] = {{0.00000000000000000000f, 1.73205077648162841797f},
                                         {1.00000000000000000000f, 1.41421353816986083984f},
                                         {1.00000000000000000000f, 1.41421353816986083984f},
                                         {1.00000000000000000000f, 1.41421353816986083984f},
                                         {1.73205077648162841797f, 0.00000000000000000000f},
                                         {1.73205077648162841797f, 3.46410155296325683594f}};

TEST(DISTANCES_TEST, EUCLIDEAN_EXCEPTION_TEST) {
    // to high-dimensionality
    ASSERT_THROW(juml::euclidean(af::constant(0, 1, 1, 3), af::constant(0, 1, 1)), std::invalid_argument);
    ASSERT_THROW(juml::euclidean(af::constant(0, 1, 1), af::constant(0, 1, 1, 3)), std::invalid_argument);
    ASSERT_THROW(juml::euclidean(af::constant(0, 1, 1, 1, 3), af::constant(0, 1, 1)), std::invalid_argument);
    ASSERT_THROW(juml::euclidean(af::constant(0, 1, 1), af::constant(0, 1, 1, 1, 3)), std::invalid_argument);

    // features count does not fit
    ASSERT_THROW(juml::euclidean(af::constant(0, 3), af::constant(0, 2)), std::invalid_argument);
}

TEST(DISTANCES_TEST, EUCLIDEAN_TEST_CPU) {
    af::setBackend(AF_BACKEND_CPU);

    af::array from(af::dim4(3, 2), reinterpret_cast<const float*>(ORIGINS));
    af::array to(af::dim4(3, 6), reinterpret_cast<const float*>(DESTINATIONS));
    af::array distances(af::dim4(2, 6), reinterpret_cast<const float*>(EUCLIDEAN_DISTANCES));

    ASSERT_EQ(af::allTrue(juml::euclidean(from, to) == distances).scalar<char>(), 1);
}

#ifdef JUML_OPENCL
TEST(DISTANCES_TEST, EUCLIDEAN_TEST_OPENCL) {
    af::setBackend(AF_BACKEND_CPU);

    af::array from(af::dim4(3, 2), reinterpret_cast<const float*>(ORIGINS));
    af::array to(af::dim4(3, 6), reinterpret_cast<const float*>(DESTINATIONS));
    af::array distances(af::dim4(2, 6), reinterpret_cast<const float*>(EUCLIDEAN_DISTANCES));

    ASSERT_EQ(af::allTrue(juml::euclidean(from, to) == distances).scalar<char>(), 1);
}
#endif // JUML_OPENCL

#ifdef JUML_CUDA
TEST(DISTANCES_TEST, EUCLIDEAN_TEST_CUDA) {
    af::setBackend(AF_BACKEND_CPU);

    af::array from(af::dim4(3, 2), reinterpret_cast<const float*>(ORIGINS));
    af::array to(af::dim4(3, 6), reinterpret_cast<const float*>(DESTINATIONS));
    af::array distances(af::dim4(2, 6), reinterpret_cast<const float*>(EUCLIDEAN_DISTANCES));

    ASSERT_EQ(af::allTrue(juml::euclidean(from, to) == distances).scalar<char>(), 1);
}
#endif // JUML_CUDA

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
