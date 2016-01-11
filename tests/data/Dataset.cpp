#include <arrayfire.h>
#include <exception>
#include <iostream>
#include <gtest/gtest.h>
#include <mpi.h>
#include <string>

#include "data/Dataset.h"
#include "util/Settings.h"

const std::string FILE_PATH = "../../../datasets/mpi_ranks.h5";
const std::string ONE_D_FLOAT = "1D_FLOAT";
const std::string TWO_D_FLOAT = "2D_FLOAT";
const std::string ONE_D_INT = "1D_INT";
const std::string TWO_D_INT = "2D_INT";

TEST (DATASET_TEST, LOAD_EQUAL_CHUNKS_1D_FLOAT_CPU_TEST) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    juml::Settings::getInstance().set_backend(AF_BACKEND_CPU);
    juml::Dataset data_1D(FILE_PATH, ONE_D_FLOAT);  
    
    data_1D.load_equal_chunks();
    for (size_t row = 0; row < data_1D.data().elements(); ++row) {
        ASSERT_FLOAT_EQ(data_1D.data().host<float>()[row], (float)rank);
    }
}

TEST (DATASET_TEST, LOAD_EQUAL_CHUNKS_2D_FLOAT_CPU_TEST) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    juml::Settings::getInstance().set_backend(AF_BACKEND_CPU);(AF_BACKEND_CPU);
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);
    
    data_2D.load_equal_chunks();
    for (size_t row = 0; row < data_2D.data().elements(); ++row) {
        ASSERT_FLOAT_EQ(data_2D.data().host<float>()[row], (float)rank);
    }
}

TEST (DATASET_TEST, LOAD_EQUAL_CHUNKS_1D_INT_CPU_TEST) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    juml::Settings::getInstance().set_backend(AF_BACKEND_CPU);  
    juml::Dataset data_1D(FILE_PATH, ONE_D_INT);
    
    data_1D.load_equal_chunks();
    for (size_t row = 0; row < data_1D.data().elements(); ++row) {
        ASSERT_EQ(data_1D.data().host<int>()[row], rank);
    }
}

TEST (DATASET_TEST, LOAD_EQUAL_CHUNKS_2D_INT_CPU_TEST) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    juml::Settings::getInstance().set_backend(AF_BACKEND_CPU);    
    juml::Dataset data_2D(FILE_PATH, TWO_D_INT);
    
    data_2D.load_equal_chunks();
    for (size_t row = 0; row < data_2D.data().elements(); ++row) {
        ASSERT_EQ(data_2D.data().host<int>()[row], rank);
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
