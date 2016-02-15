#include <arrayfire.h>
#include <exception>
#include <ios>
#include <iostream>
#include <gtest/gtest.h>
#include <mpi.h>
#include <string>

#include "data/Dataset.h"

const std::string FILE_PATH   = "../../../datasets/mpi_ranks.h5";
const std::string ONE_D_FLOAT = "1D_FLOAT";
const std::string TWO_D_FLOAT = "2D_FLOAT";
const std::string ONE_D_INT   = "1D_INT";
const std::string TWO_D_INT   = "2D_INT";

class DATASET_TEST : public testing::Test
{
public:
    int rank_;

    DATASET_TEST() {
        MPI_Comm_rank(MPI_COMM_WORLD, &this->rank_);
    }
};

const std::string FILE_PATH_ROWNUMBER = "../../../datasets/rownumInColumns5x3.h5";
const std::string ROWNUMBER_SETNAME = "testset";

TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_SINGLE_PROCESS) {
    if (rank_ != 0) return;
    af::setBackend(AF_BACKEND_CPU);
    juml::Dataset data(FILE_PATH_ROWNUMBER, ROWNUMBER_SETNAME, MPI_COMM_SELF);

    data.load_equal_chunks();
    // this is currently not the case ! the matrix is read as 5 x 3
    //file contains 5 rows and 3 columns. We should read it as 5 columns and 3 rows
    ASSERT_EQ(5, data.data().dims(0)) << "Number of Columns in File does not match number of Rows in Dataset";
    ASSERT_EQ(3, data.data().dims(1)) << "Number of Rows in File does not match number of Columns in Dataset";

    af::print("data", data.data());
    for (int row = 0; row < 5; ++row) {
        af::print("row created", af::transpose(af::constant(row, 3, s32)));
        ASSERT_TRUE(af::sum<int>(data.data().row(row) != af::transpose(af::constant(row, 3, s32))) == 0) << "row " << row << " does not only contain the row number";
    }
}

TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_1D_FLOAT_CPU_TEST) {
    af::setBackend(AF_BACKEND_CPU);
    juml::Dataset data_1D(FILE_PATH, ONE_D_FLOAT);

    data_1D.load_equal_chunks();
    for (size_t row = 0; row < data_1D.data().elements(); ++row) {
        ASSERT_FLOAT_EQ(data_1D.data()(row).scalar<float>(), (float)this->rank_);
    }
}

TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_2D_FLOAT_CPU_TEST) {
    af::setBackend(AF_BACKEND_CPU);
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);

    data_2D.load_equal_chunks();
    for (size_t row = 0; row < data_2D.data().elements(); ++row) {
        ASSERT_FLOAT_EQ(data_2D.data()(row).scalar<float>(), (float)this->rank_);
    }
}

TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_1D_INT_CPU_TEST) {
    af::setBackend(AF_BACKEND_CPU);
    juml::Dataset data_1D(FILE_PATH, ONE_D_INT);

    data_1D.load_equal_chunks();
    for (size_t row = 0; row < data_1D.data().elements(); ++row) {
        ASSERT_EQ(data_1D.data()(row).scalar<int>(), this->rank_);
    }
}

TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_2D_INT_CPU_TEST) {
    af::setBackend(AF_BACKEND_CPU);
    juml::Dataset data_2D(FILE_PATH, TWO_D_INT);

    data_2D.load_equal_chunks();
    for (size_t row = 0; row < data_2D.data().elements(); ++row) {
        ASSERT_EQ(data_2D.data().device<int>()[row], this->rank_);
    }
}

#ifdef JUML_OPENCL
TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_1D_FLOAT_OPENCL_TEST) {
    af::setBackend(AF_BACKEND_OPENCL);
    juml::Dataset data_1D(FILE_PATH, ONE_D_FLOAT);

    data_1D.load_equal_chunks();
    float* data_ptr =  data_1D.data().host<float>();
    for (size_t row = 0; row < data_1D.data().elements(); ++row) {
        ASSERT_FLOAT_EQ(data_ptr[row], (float)this->rank_);
    }
}

TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_2D_FLOAT_OPENCL_TEST) {
    af::setBackend(AF_BACKEND_OPENCL);
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);

    data_2D.load_equal_chunks();
    float* data_ptr =  data_2D.data().host<float>();
    for (size_t row = 0; row < data_2D.data().elements(); ++row) {
        ASSERT_FLOAT_EQ(data_ptr[row], (float)this->rank_);
    }
}

TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_1D_INT_OPENCL_TEST) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    af::setBackend(AF_BACKEND_OPENCL);
    juml::Dataset data_1D(FILE_PATH, ONE_D_INT);

    data_1D.load_equal_chunks();
    int* data_ptr =  data_1D.data().host<int>();
    for (size_t row = 0; row < data_1D.data().elements(); ++row) {
        ASSERT_EQ(data_ptr[row], this->rank_);
    }
}

TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_2D_INT_OPENCL_TEST) {
    af::setBackend(AF_BACKEND_OPENCL);
    juml::Dataset data_2D(FILE_PATH, TWO_D_INT);

    data_2D.load_equal_chunks();

    int* data_ptr =  data_2D.data().host<int>();
    for (size_t row = 0; row < data_2D.data().elements(); ++row) {
        ASSERT_EQ(data_ptr[row], this->rank_);
    }
}
#endif
// OPENCL

#ifdef JUML_CUDA
TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_1D_FLOAT_CUDA_TEST) {
    af::setBackend(AF_BACKEND_CUDA);
    juml::Dataset data_1D(FILE_PATH, ONE_D_FLOAT);

    data_1D.load_equal_chunks();
    float* data_ptr =  data_1D.data().host<float>();
    for (size_t row = 0; row < data_1D.data().elements(); ++row) {
        ASSERT_FLOAT_EQ(data_ptr[row], (float)this->rank_);
    }
}

TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_2D_FLOAT_CUDA_TEST) {
    af::setBackend(AF_BACKEND_CUDA);
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);

    data_2D.load_equal_chunks();
    float* data_ptr =  data_2D.data().host<float>();
    for (size_t row = 0; row < data_2D.data().elements(); ++row) {
        ASSERT_FLOAT_EQ(data_ptr[row], (float)this->rank_);
    }
}

TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_1D_INT_CUDA_TEST) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    af::setBackend(AF_BACKEND_CUDA);
    juml::Dataset data_1D(FILE_PATH, ONE_D_INT);

    data_1D.load_equal_chunks();
    int* data_ptr =  data_1D.data().host<int>();
    for (size_t row = 0; row < data_1D.data().elements(); ++row) {
        ASSERT_EQ(data_ptr[row], this->rank_);
    }
}

TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_2D_INT_CUDA_TEST) {
    af::setBackend(AF_BACKEND_CUDA);
    juml::Dataset data_2D(FILE_PATH, TWO_D_INT);

    data_2D.load_equal_chunks();

    int* data_ptr =  data_2D.data().host<int>();
    for (size_t row = 0; row < data_2D.data().elements(); ++row) {
        ASSERT_EQ(data_ptr[row], this->rank_);
    }
}
#endif
// CUDA

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
