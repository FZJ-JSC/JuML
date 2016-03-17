#include <exception>
#include <gtest/gtest.h>
#include <iostream>
#include <mpi.h>
#include <string>

#include "data/Dataset.h"
#include "core/Backend.h"
#include "classification/CART.h"

static const std::string FILE_PATH = JUML_DATASETS"/iris.h5";
static const std::string SAMPLES = "samples";
static const std::string LABELS = "labels";

TEST (CART_TEST, IRIS_CPU_TEST) {
    juml::CART cart(juml::Backend::CPU);
    juml::Dataset X(FILE_PATH, SAMPLES);
    juml::Dataset y(FILE_PATH, LABELS);

    cart.fit(X, y);

    ASSERT_EQ(true, false);
}

int main(int argc, char** argv) {
    int result = -1;
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ::testing::InitGoogleTest(&argc, argv);

    // suppress output from the other ranks
    if (rank > 0) {
        ::testing::UnitTest& unit_test = *::testing::UnitTest::GetInstance();
        ::testing::TestEventListeners& listeners = unit_test.listeners();
        delete listeners.Release(listeners.default_result_printer());
        listeners.Append(new ::testing::EmptyTestEventListener);
    }

    try {
        result = RUN_ALL_TESTS();
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
    }
    MPI_Finalize();

    return result;
}
