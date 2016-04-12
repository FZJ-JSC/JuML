#include <arrayfire.h>
#include <cstdio>
#include <exception>
#include <iostream>
#include <gtest/gtest.h>
#include <mpi.h>
#include <string>

#include "core/Backend.h"
#include "core/HDF5.h"
#include "core/Test.h"
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


TEST_ALL_F (HDF5_TEST, WRITE_ARRAY1D_TEST) {
    if (rank_ != 0) return;
    af::array data = af::constant(1, 4);
    hid_t file_id = juml::hdf5::open_file(WRITE_FILE);
    juml::hdf5::write_array(file_id, TEST_SET, data);
    juml::hdf5::close_file(file_id);

    juml::Dataset loaded(WRITE_FILE, TEST_SET, MPI_COMM_SELF);
    loaded.load_equal_chunks();
    if (rank_ == 0) {
        std::remove(WRITE_FILE.c_str());
    }
    ASSERT_TRUE(af::allTrue<bool>(loaded.data() == data));
}

TEST_ALL_F (HDF5_TEST, WRITE_ARRAY2D_TEST) {
    if (rank_ != 0) return;
    af::array data = af::constant(1, 4, 6);
    hid_t file_id = juml::hdf5::open_file(WRITE_FILE);
    juml::hdf5::write_array(file_id, TEST_SET, data);
    juml::hdf5::close_file(file_id);

    juml::Dataset loaded(WRITE_FILE, TEST_SET, MPI_COMM_SELF);
    loaded.load_equal_chunks();
    if (rank_ == 0) {
        std::remove(WRITE_FILE.c_str());
    }
    ASSERT_TRUE(af::allTrue<bool>(loaded.data() == data));
}

TEST_ALL_F (HDF5_TEST, WRITE_ARRAY3D_TEST) {
    if (rank_ != 0) return;
    af::array data = af::constant(1, 4, 6, 5);
    hid_t file_id = juml::hdf5::open_file(WRITE_FILE);
    juml::hdf5::write_array(file_id, TEST_SET, data);
    juml::hdf5::close_file(file_id);

    juml::Dataset loaded(WRITE_FILE, TEST_SET, MPI_COMM_SELF);
    loaded.load_equal_chunks();
    if (rank_ == 0) {
        std::remove(WRITE_FILE.c_str());
    }
    ASSERT_TRUE(af::allTrue<bool>(loaded.data() == data));
}

TEST_ALL_F (HDF5_TEST, PREAD_ARRAY_TEST) {
    af::array data = af::constant(1, 4, 6);
    if (rank_ == 0) {

        hid_t file_id = juml::hdf5::open_file(WRITE_FILE);
        juml::hdf5::write_array(file_id, TEST_SET, data);
        juml::hdf5::close_file(file_id);
    }

    hid_t in_file = juml::hdf5::popen_file(WRITE_FILE, MPI_COMM_WORLD);
    ASSERT_GT(in_file, 0);
    af::array loaded = juml::hdf5::pread_array(in_file, TEST_SET);
    ASSERT_TRUE(af::allTrue<bool>(loaded == data));
    juml::hdf5::close_file(in_file);
    if (rank_ == 0) {
        std::remove(WRITE_FILE.c_str());
    }
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