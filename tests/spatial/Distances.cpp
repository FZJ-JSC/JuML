#include <arrayfire.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <armadillo>

#include "core/Backend.h"
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
                                         {1.00000000000000000000f, 1.41421365737915039062f},
                                         {1.00000000000000000000f, 1.41421365737915039062f},
                                         {1.00000000000000000000f, 1.41421365737915039062f},
                                         {1.73205077648162841797f, 0.00000000000000000000f},
                                         {1.73205077648162841797f, 3.46410155296325683594f}};

const float MANHATTAN_DISTANCES[6][2] = {{0.0f, 3.0f},
                                         {1.0f, 2.0f},
                                         {1.0f, 2.0f},
                                         {1.0f, 2.0f},
                                         {3.0f, 0.0f},
                                         {3.0f, 6.0f}};

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
    juml::Backend::set(juml::Backend::CPU);

    af::array from(af::dim4(3, 2), reinterpret_cast<const float*>(ORIGINS));
    af::array to(af::dim4(3, 6), reinterpret_cast<const float*>(DESTINATIONS));
    af::array distances(af::dim4(2, 6), reinterpret_cast<const float*>(EUCLIDEAN_DISTANCES));

    ASSERT_EQ(af::allTrue(juml::euclidean(from, to) == distances).scalar<char>(), 1);
}

#ifdef JUML_OPENCL
TEST(DISTANCES_TEST, EUCLIDEAN_TEST_OPENCL) {
    juml::Backend::set(juml::Backend::OPENCL);

    af::array from(af::dim4(3, 2), reinterpret_cast<const float*>(ORIGINS));
    af::array to(af::dim4(3, 6), reinterpret_cast<const float*>(DESTINATIONS));
    af::array distances = juml::euclidean(from, to);

    for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 2; ++col) {
            ASSERT_FLOAT_EQ(distances(col, row).scalar<float>(), EUCLIDEAN_DISTANCES[row][col]);
        }
    }
}
#endif // JUML_OPENCL

#ifdef JUML_CUDA
TEST(DISTANCES_TEST, EUCLIDEAN_TEST_CUDA) {
    juml::Backend::set(juml::Backend::CUDA);

    af::array from(af::dim4(3, 2), reinterpret_cast<const float*>(ORIGINS));
    af::array to(af::dim4(3, 6), reinterpret_cast<const float*>(DESTINATIONS));
    af::array distances(af::dim4(2, 6), reinterpret_cast<const float*>(EUCLIDEAN_DISTANCES));

    ASSERT_EQ(af::allTrue(juml::euclidean(from, to) == distances).scalar<char>(), 1);
}
#endif // JUML_CUDA

TEST(DISTANCES_TEST, MANHATTAN_EXCEPTION_TEST) {
    // to high-dimensionality
    ASSERT_THROW(juml::manhattan(af::constant(0, 1, 1, 3), af::constant(0, 1, 1)), std::invalid_argument);
    ASSERT_THROW(juml::manhattan(af::constant(0, 1, 1), af::constant(0, 1, 1, 3)), std::invalid_argument);
    ASSERT_THROW(juml::manhattan(af::constant(0, 1, 1, 1, 3), af::constant(0, 1, 1)), std::invalid_argument);
    ASSERT_THROW(juml::manhattan(af::constant(0, 1, 1), af::constant(0, 1, 1, 1, 3)), std::invalid_argument);

    // features count does not fit
    ASSERT_THROW(juml::manhattan(af::constant(0, 3), af::constant(0, 2)), std::invalid_argument);
}

TEST(DISTANCES_TEST, MANHATTAN_TEST_CPU) {
    juml::Backend::set(juml::Backend::CPU);

    af::array from(af::dim4(3, 2), reinterpret_cast<const float*>(ORIGINS));
    af::array to(af::dim4(3, 6), reinterpret_cast<const float*>(DESTINATIONS));
    af::array distances(af::dim4(2, 6), reinterpret_cast<const float*>(MANHATTAN_DISTANCES));

    ASSERT_EQ(af::allTrue(juml::manhattan(from, to) == distances).scalar<char>(), 1);
}

#ifdef JUML_OPENCL
TEST(DISTANCES_TEST, MANHATTAN_TEST_OPENCL) {
    juml::Backend::set(juml::Backend::OPENCL);

    af::array from(af::dim4(3, 2), reinterpret_cast<const float*>(ORIGINS));
    af::array to(af::dim4(3, 6), reinterpret_cast<const float*>(DESTINATIONS));
    af::array distances(af::dim4(2, 6), reinterpret_cast<const float*>(MANHATTAN_DISTANCES));

    ASSERT_EQ(af::allTrue(juml::manhattan(from, to) == distances).scalar<char>(), 1);
}
#endif // JUML_OPENCL

#ifdef JUML_CUDA
TEST(DISTANCES_TEST, MANHATTAN_TEST_CUDA) {
    juml::Backend::set(juml::Backend::CUDA);

    af::array from(af::dim4(3, 2), reinterpret_cast<const float*>(ORIGINS));
    af::array to(af::dim4(3, 6), reinterpret_cast<const float*>(DESTINATIONS));
    af::array distances(af::dim4(2, 6), reinterpret_cast<const float*>(MANHATTAN_DISTANCES));

    ASSERT_EQ(af::allTrue(juml::manhattan(from, to) == distances).scalar<char>(), 1);
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
