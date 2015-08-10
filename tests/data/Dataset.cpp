#include <armadillo>
#include <exception>
#include <iostream>
#include <gtest/gtest.h>
#include <mpi.h>
#include <string>

#include "data/Dataset.h"

const std::string FILE_PATH = "../../../datasets/mpi_ranks.h5";
const std::string ONE_D = "1D";
const std::string TWO_D = "2D";

TEST (DATASET_TEST, LOAD_EQUAL_CHUNKS_1D_TEST) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    juml::Dataset<float> data_1D(FILE_PATH, ONE_D);
    data_1D.load_equal_chunks();
    for (size_t row = 0; row < data_1D.data().n_elem; ++row) {
        ASSERT_FLOAT_EQ(data_1D.data()[row], (float)rank);
    }
}


TEST (DATASET_TEST, LOAD_EQUAL_CHUNKS_2D_TEST) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    juml::Dataset<float> data_2D(FILE_PATH, TWO_D);
    data_2D.load_equal_chunks();

    for (size_t row = 0; row < data_2D.data().n_elem; ++row) {
        ASSERT_FLOAT_EQ(data_2D.data()[row], (float)rank);
    }
}

int main(int argc, char** argv) {
    int result = -1;
    
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    try {
        result = RUN_ALL_TESTS();
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
    }
    MPI_Finalize();

    return result;
}
