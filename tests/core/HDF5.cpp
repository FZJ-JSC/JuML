#include <arrayfire.h>
#include <cstdio>
#include <exception>
#include <iostream>
#include <gtest/gtest.h>
#include <mpi.h>
#include <string>

#include "core/Backend.h"
#include "core/HDF5.h"
#include "data/Dataset.h"

static const std::string WRITE_FILE = "write_test.h5";
static const std::string TEST_SET = "test_set";

class HDF5_TEST : public testing::Test
{
public:
    int rank_;
    int size_;

    HDF5_TEST() {
        MPI_Comm_rank(MPI_COMM_WORLD, &this->rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &this->size_);
    }
};


TEST_F (HDF5_TEST, WRITE_ARRAY_TEST) {
    juml::Backend::set(juml::Backend::CPU);
    if (rank_ != 0) return;
    af::array data = af::constant(1, 3, 4);
    af_print(data);
    hid_t file_id = juml::hdf5::create_file(WRITE_FILE);
    juml::hdf5::write_array(file_id, TEST_SET, data);
    juml::hdf5::close_file(file_id);

    juml::Dataset loaded(WRITE_FILE, TEST_SET, MPI_COMM_SELF);
    ASSERT_TRUE(af::allTrue<bool>(loaded.data() == data));
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