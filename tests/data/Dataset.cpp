#include <arrayfire.h>
#include <cstdio>
#include <exception>
#include <iostream>
#include <gtest/gtest.h>
#include <mpi.h>
#include <string>

#include "core/Test.h"
#include "data/Dataset.h"

const std::string FILE_PATH   = JUML_DATASETS"/mpi_ranks.h5";
const std::string ONE_D_FLOAT = "1D_FLOAT";
const std::string TWO_D_FLOAT = "2D_FLOAT";
const std::string ONE_D_INT   = "1D_INT";
const std::string TWO_D_INT   = "2D_INT";

const std::string FILE_PATH_ROWNUMBER = JUML_DATASETS"/rownumInColumns5x3.h5";
const std::string ROWNUMBER_SETNAME = "testset";

const std::string DUMP_FILE    = "dumpTest.h5";
const std::string DUMP_DATASET = "DUMPED";
const std::string DUMP_DATASET2 = "TEST_DUMPED";

class DATASET_TEST : public testing::Test
{
public:
    int rank_;
    int size_;

    DATASET_TEST() {
        MPI_Comm_rank(MPI_COMM_WORLD, &this->rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &this->size_);
    }
};

TEST_ALL_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_SINGLE_PROCESS) {
    if (this->rank_ != 0) return;
    juml::Dataset data(FILE_PATH_ROWNUMBER, ROWNUMBER_SETNAME, MPI_COMM_SELF);

    data.load_equal_chunks();
    ASSERT_EQ(3, data.data().dims(0)) << "Number of Columns in File does not match number of Rows in Dataset";
    ASSERT_EQ(5, data.data().dims(1)) << "Number of Rows in File does not match number of Columns in Dataset";

    for (int col = 0; col < 5; ++col) {
        ASSERT_TRUE(af::sum<int>(data.data().col(col) != af::constant(col, 3, s32)) == 0) << "col " << col << " does not only contain the row number";
    }
}

TEST_ALL_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_1D_FLOAT) {
    juml::Dataset data_1D(FILE_PATH, ONE_D_FLOAT);
    data_1D.load_equal_chunks();

    for (size_t col = 0; col < data_1D.data().dims(1); ++col) {
        ASSERT_TRUE(af::allTrue<bool>(data_1D.data().col(col) == (float)this->rank_));
    }
}

TEST_ALL_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_2D_FLOAT) {
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);
    data_2D.load_equal_chunks();
    for (size_t col = 0; col < data_2D.data().dims(1); ++col) {
        ASSERT_TRUE(af::allTrue<bool>(data_2D.data().col(col) == (float)this->rank_));
    }
}

TEST_ALL_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_1D_INT) {
    juml::Dataset data_1D(FILE_PATH, ONE_D_INT);
    data_1D.load_equal_chunks();
    for (size_t col = 0; col < data_1D.data().dims(1); ++col) {
        ASSERT_TRUE(af::allTrue<bool>(data_1D.data().col(col) == this->rank_));
    }
}

TEST_ALL_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_2D_INT) {
    juml::Dataset data_2D(FILE_PATH, TWO_D_INT);
    data_2D.load_equal_chunks();
    for (size_t col = 0; col < data_2D.data().dims(1); ++col) {
	    ASSERT_TRUE(af::allTrue<bool>(data_2D.data().col(col) == this->rank_));
    }
}

TEST_ALL_F(DATASET_TEST, DUMP_EQUAL_CHUNKS) {
    juml::Backend::set(juml::Backend::CPU);
    af::array data = af::constant(rank_, 2, 10, 4, s32);
    juml::Dataset dataset(data, MPI_COMM_WORLD);
    dataset.dump_equal_chunks(DUMP_FILE, DUMP_DATASET);

    juml::Dataset loaded(DUMP_FILE, DUMP_DATASET);
    loaded.load_equal_chunks();
    if (rank_ == 0) {
        std::remove(DUMP_FILE.c_str());
    }
    ASSERT_TRUE(af::allTrue<bool>(loaded.data() == data));
}

TEST_ALL_F(DATASET_TEST, DUMP_EQUAL_CHUNKS_APPEND) {
    juml::Backend::set(juml::Backend::CPU);
    af::array data1 = af::constant(rank_, 2, 3, s32);
    juml::Dataset dataset1(data1, MPI_COMM_WORLD);
    dataset1.dump_equal_chunks(DUMP_FILE, DUMP_DATASET);

    af::array data2 = af::constant(rank_ * -1, 2, 4, s32);
    juml::Dataset dataset2(data2, MPI_COMM_WORLD);
    dataset2.dump_equal_chunks(DUMP_FILE, DUMP_DATASET2);
    juml::Dataset loaded(DUMP_FILE, DUMP_DATASET);
    juml::Dataset loaded2(DUMP_FILE, DUMP_DATASET2);
    loaded.load_equal_chunks();
    loaded2.load_equal_chunks();
    if (rank_ == 0) {
        std::remove(DUMP_FILE.c_str());
    }
    ASSERT_TRUE(af::allTrue<bool>(loaded.data() == data1));
    ASSERT_TRUE(af::allTrue<bool>(loaded2.data() == data2));
}

TEST_ALL_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_PREVENT_RELOAD) {
    juml::Dataset data_1D(FILE_PATH, ONE_D_INT);
    time_t loading_time = data_1D.loading_time();
    ASSERT_EQ(loading_time, 0);
    data_1D.load_equal_chunks();
    ASSERT_GT(data_1D.loading_time(), loading_time);
}

TEST_ALL_F(DATASET_TEST, CREATE_FROM_ARRAY) {
    af::array data = af::constant(1, 4, 4);
    juml::Dataset set(data);
    ASSERT_TRUE(af::allTrue<bool>(set.data() == data));

    // call load_equal_chunks, nothing is done
    set.load_equal_chunks();
}

TEST_ALL_F(DATASET_TEST, NORMALIZE_0_TO_1_DEP_ALL) {
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);
    data_2D.load_equal_chunks();
    data_2D.normalize();
    for (size_t row = 0; row < data_2D.data().dims(0); ++row)
        for (size_t col = 0; col < data_2D.data().dims(1); ++col)
            ASSERT_FLOAT_EQ((float)this->rank_/3.0, data_2D.data()(row,col).scalar<float>());
}

TEST_ALL_F(DATASET_TEST, NORMALIZE_1_TO_1_DEP_ALL) {
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);
    data_2D.load_equal_chunks();
    data_2D.normalize(1,1);
    for (size_t row = 0; row < data_2D.data().dims(0); ++row)
        for (size_t col = 0; col < data_2D.data().dims(1); ++col)
            ASSERT_FLOAT_EQ(1.0, data_2D.data()(row,col).scalar<float>());
}

TEST_ALL_F(DATASET_TEST, NORMALIZE_0_TO_1_INDEP_ALL) {
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);
    data_2D.load_equal_chunks();
    data_2D.normalize(0,1,true);
    for (size_t row = 0; row < data_2D.data().dims(0); ++row)
        for (size_t col = 0; col < data_2D.data().dims(1); ++col)
            ASSERT_FLOAT_EQ((float)this->rank_/3.0, data_2D.data()(row,col).scalar<float>());
}

TEST_ALL_F(DATASET_TEST, NORMALIZE_MINUS20_TO_10_INDEP_ALL) {
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);
    data_2D.load_equal_chunks();
    data_2D.normalize(-20,10,true);
    for (size_t row = 0; row < data_2D.data().dims(0); ++row)
        for (size_t col = 0; col < data_2D.data().dims(1); ++col)
            ASSERT_FLOAT_EQ(10.0*(float)this->rank_ - 20.0, data_2D.data()(row,col).scalar<float>());
}

TEST_ALL_F(DATASET_TEST, NORMALIZE_MINUS20_TO_10_INDEP_1_2) {
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);
    data_2D.load_equal_chunks();
    data_2D.normalize(-20, 10, true, af::range(2));
    for (size_t row = 0; row < data_2D.data().dims(0); ++row) {
        for (size_t col = 0; col < data_2D.data().dims(1); ++col) {
            if (row == 2)
                ASSERT_FLOAT_EQ((float) this->rank_, data_2D.data()(row, col).scalar<float>());
            else
                ASSERT_FLOAT_EQ(10.0 * (float) this->rank_ - 20.0, data_2D.data()(row, col).scalar<float>());
        }
    }
}

TEST_ALL_F(DATASET_TEST, NORMALIZE_STD_1_DEP_ALL) {
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);
    data_2D.load_equal_chunks();
    data_2D.normalize_stddev(1.0);
    af::array std = data_2D.stddev();
    for (size_t row = 0; row < std.dims(0); ++row)
        ASSERT_FLOAT_EQ(1.0, std(row).scalar<float>());
}

TEST_ALL_F(DATASET_TEST, NORMALIZE_STD_2_INDEP_ALL) {
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);
    data_2D.load_equal_chunks();
    data_2D.normalize_stddev(2.0, true);
    af::array std = data_2D.stddev();
    for (size_t row = 0; row < std.dims(0); ++row)
        ASSERT_FLOAT_EQ(2.0, std(row).scalar<float>());
}

TEST_ALL_F(DATASET_TEST, NORMALIZE_STD_3_INDEP_1_2) {
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);
    data_2D.load_equal_chunks();
    data_2D.normalize_stddev(3.0, true, af::range(2));
    af::array std = data_2D.stddev();
    for (size_t row = 0; row < std.dims(0); ++row)
        if(row != 2)
            ASSERT_FLOAT_EQ(3.0, std(row).scalar<float>());
        else
            ASSERT_FLOAT_EQ(1.0671874, std(row).scalar<float>());
}

TEST_ALL_F(DATASET_TEST, MEAN_ALL) {
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);
    data_2D.load_equal_chunks();
    af::array mean = data_2D.mean(true);
    ASSERT_FLOAT_EQ(mean.scalar<float>(), 21.0/18.0);
}

TEST_ALL_F(DATASET_TEST, MEAN) {
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);
    data_2D.load_equal_chunks();
    af::array mean = data_2D.mean();
    for (size_t col=0; col < mean.dims(0); col++)
        ASSERT_FLOAT_EQ(mean(col).scalar<float>(), 7.0/6.0);
}

TEST_ALL_F(DATASET_TEST, STDDEV) {
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);
    data_2D.load_equal_chunks();
    af::array stddev = data_2D.stddev();
    for (size_t col=0; col < stddev.dims(0); col++)
        ASSERT_FLOAT_EQ(stddev(col).scalar<float>(), 1.0671873);
}

TEST_ALL_F(DATASET_TEST, STDDEV_ALL) {
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);
    data_2D.load_equal_chunks();
    af::array stddev = data_2D.stddev(true);
    ASSERT_FLOAT_EQ(stddev.scalar<float>(), 1.0671873);
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
