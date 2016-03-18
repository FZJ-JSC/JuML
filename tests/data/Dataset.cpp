#include <arrayfire.h>
#include <cstdio>
#include <exception>
#include <iostream>
#include <gtest/gtest.h>
#include <mpi.h>
#include <string>

#include "core/Backend.h"
#include "data/Dataset.h"

const std::string FILE_PATH   = JUML_DATASETS"/mpi_ranks.h5";
const std::string ONE_D_FLOAT = "1D_FLOAT";
const std::string TWO_D_FLOAT = "2D_FLOAT";
const std::string ONE_D_INT   = "1D_INT";
const std::string TWO_D_INT   = "2D_INT";

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

const std::string FILE_PATH_ROWNUMBER = JUML_DATASETS"/rownumInColumns5x3.h5";
const std::string ROWNUMBER_SETNAME = "testset";
const std::string DUMP_FILE = "dump_test.h5";

TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_SINGLE_PROCESS) {
    if (rank_ != 0) return;
    juml::Backend::set(juml::Backend::CPU);
    juml::Dataset data(FILE_PATH_ROWNUMBER, ROWNUMBER_SETNAME, MPI_COMM_SELF);

    data.load_equal_chunks();
    ASSERT_EQ(3, data.data().dims(0)) << "Number of Columns in File does not match number of Rows in Dataset";
    ASSERT_EQ(5, data.data().dims(1)) << "Number of Rows in File does not match number of Columns in Dataset";

    for (int col = 0; col < 5; ++col) {
        ASSERT_TRUE(af::sum<int>(data.data().col(col) != af::constant(col, 3, s32)) == 0) << "col " << col << " does not only contain the row number";
    }
}

TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_1D_FLOAT_CPU_TEST) {
    juml::Backend::set(juml::Backend::CPU);
    juml::Dataset data_1D(FILE_PATH, ONE_D_FLOAT);

    data_1D.load_equal_chunks();
    for (size_t col = 0; col < data_1D.data().dims(1); ++col) {
        ASSERT_TRUE(af::allTrue<bool>(data_1D.data().col(col) == (float)this->rank_));
    }
}

TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_2D_FLOAT_CPU_TEST) {
    juml::Backend::set(juml::Backend::CPU);
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);

    data_2D.load_equal_chunks();
    for (size_t col = 0; col < data_2D.data().dims(1); ++col) {
        ASSERT_TRUE(af::allTrue<bool>(data_2D.data().col(col) == (float)this->rank_));
    }
}

TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_1D_INT_CPU_TEST) {
    juml::Backend::set(juml::Backend::CPU);
    juml::Dataset data_1D(FILE_PATH, ONE_D_INT);

    data_1D.load_equal_chunks();
    for (size_t col = 0; col < data_1D.data().dims(1); ++col) {
        ASSERT_TRUE(af::allTrue<bool>(data_1D.data().col(col) == this->rank_));
    }
}

TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_2D_INT_CPU_TEST) {
    juml::Backend::set(juml::Backend::CPU);
    juml::Dataset data_2D(FILE_PATH, TWO_D_INT);

    data_2D.load_equal_chunks();
    for (size_t col = 0; col < data_2D.data().dims(1); ++col) {
	ASSERT_TRUE(af::allTrue<bool>(data_2D.data().col(col) == this->rank_));
    }
}

TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_PREVENT_RELOAD_CPU_TEST) {
    juml::Backend::set(juml::Backend::CPU);
    juml::Dataset data_1D(FILE_PATH, ONE_D_INT);
    time_t loading_time = data_1D.loading_time();
    ASSERT_EQ(loading_time, 0);
    data_1D.load_equal_chunks();
    ASSERT_GT(data_1D.loading_time(), loading_time);
}

TEST_F(DATASET_TEST, CREATE_FROM_ARRAY) {
    juml::Backend::set(juml::Backend::CPU);
    af::array data = af::constant(1, 4, 4);

    juml::Dataset set(data);
    ASSERT_TRUE(af::allTrue<bool>(set.data() == data));

    // call load_equal_chunks, nothing is done
    set.load_equal_chunks();
}


TEST_F(DATASET_TEST, NORMALIZE_0_TO_1_DEP_ALL) {
    juml::Backend::set(juml::Backend::CPU);
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);
    data_2D.load_equal_chunks();
    data_2D.normalize();
    for (size_t col = 0; col < data_2D.data().dims(1); ++col) {
        ASSERT_TRUE(af::allTrue<bool>(data_2D.data().col(col) == (float)this->rank_/(float)this->size_));
    }
}
TEST_F(DATASET_TEST, NORMALIZE_1_TO_1_DEP_ALL) {
    juml::Backend::set(juml::Backend::CPU);
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);
    data_2D.load_equal_chunks();
    data_2D.normalize(1,1);
    for (size_t col = 0; col < data_2D.data().dims(1); ++col) {
        ASSERT_TRUE(af::allTrue<bool>(data_2D.data().col(col) == 1));
    }
}

TEST_F(DATASET_TEST, NORMALIZE_0_TO_1_INDEP_ALL) {
    juml::Backend::set(juml::Backend::CPU);
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);
    data_2D.load_equal_chunks();
    data_2D.normalize(0,1,true);
    for (size_t col = 0; col < data_2D.data().dims(1); ++col) {
        ASSERT_TRUE(af::allTrue<bool>(data_2D.data().col(col) == (float)this->rank_/(float)this->size_));
    }
}
TEST_F(DATASET_TEST, NORMALIZE_MINUS20_TO_10_INDEP_ALL) {
    juml::Backend::set(juml::Backend::CPU);
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);
    data_2D.load_equal_chunks();
    data_2D.normalize(-20,10,true);
    for (size_t col = 0; col < data_2D.data().dims(1); ++col) {
        ASSERT_TRUE(af::allTrue<bool>(data_2D.data().col(col) == 30*(float)this->rank_/(float)this->size_ - 20));
    }
}
TEST_F(DATASET_TEST, NORMALIZE_MINUS20_TO_10_INDEP_1_2) {
    juml::Backend::set(juml::Backend::CPU);
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);
    data_2D.load_equal_chunks();
    data_2D.normalize(-20, 10, true, af::range(af::dim4(3)) < 3);
    for (size_t col = 0; col < data_2D.data().dims(1); ++col) {
        if (this->rank_ == 3)
            ASSERT_TRUE(af::allTrue<bool>(data_2D.data().col(col) == (float)this->rank_));
        else
            ASSERT_TRUE(af::allTrue<bool>(data_2D.data().col(col) == 30*(float)this->rank_/(float)this->size_ - 20));
    }
}

TEST_F(DATASET_TEST, DUMP_EQUAL_CHUNKS_TEST) {
    juml::Backend::set(juml::Backend::CPU);
    af::array data = af::constant(rank_, 2, 2);
    juml::Dataset dataset(data, MPI_COMM_WORLD);
    dataset.dump_equal_chunks(DUMP_FILE, "dataset");

    juml::Dataset loaded(DUMP_FILE,"dataset");
    loaded.load_equal_chunks();
    af::allTrue<bool>(loaded.data() == data);
    if (rank_ == 0) {
        std::remove(DUMP_FILE.c_str());
    }
}


#ifdef JUML_OPENCL
TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_1D_FLOAT_OPENCL_TEST) {
    juml::Backend::set(juml::Backend::OPENCL);
    juml::Dataset data_1D(FILE_PATH, ONE_D_FLOAT);

    data_1D.load_equal_chunks();
    for (size_t col = 0; col < data_1D.data().dims(1); ++col) {
        ASSERT_TRUE(af::allTrue<bool>(data_1D.data().col(col) == (float)this->rank_));
    }
}

TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_2D_FLOAT_OPENCL_TEST) {
    juml::Backend::set(juml::Backend::OPENCL);
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);

    data_2D.load_equal_chunks();
    for (size_t col = 0; col < data_2D.data().dims(1); ++col) {
        ASSERT_TRUE(af::allTrue<bool>(data_2D.data().col(col) == (float)this->rank_));
    }
}

TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_1D_INT_OPENCL_TEST) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    juml::Backend::set(juml::Backend::OPENCL);
    juml::Dataset data_1D(FILE_PATH, ONE_D_INT);

    data_1D.load_equal_chunks();
    for (size_t col = 0; col < data_1D.data().dims(1); ++col) {
        ASSERT_TRUE(af::allTrue<bool>(data_1D.data().col(col) == this->rank_));
    }

}

TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_2D_INT_OPENCL_TEST) {
    juml::Backend::set(juml::Backend::OPENCL);
    juml::Dataset data_2D(FILE_PATH, TWO_D_INT);

    data_2D.load_equal_chunks();

    for (size_t col = 0; col < data_2D.data().dims(1); ++col) {
        ASSERT_TRUE(af::allTrue<bool>(data_2D.data().col(col) == this->rank_));
    }
}
#endif
// OPENCL

#ifdef JUML_CUDA
TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_1D_FLOAT_CUDA_TEST) {
    juml::Backend::set(juml::Backend::CUDA);
    juml::Dataset data_1D(FILE_PATH, ONE_D_FLOAT);

    data_1D.load_equal_chunks();
    for (size_t col = 0; col < data_1D.data().dims(1); ++col) {
        ASSERT_TRUE(af::allTrue<bool>(data_1D.data().col(col) == (float)this->rank_));
    }
}

TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_2D_FLOAT_CUDA_TEST) {
    juml::Backend::set(juml::Backend::CUDA);
    juml::Dataset data_2D(FILE_PATH, TWO_D_FLOAT);

    data_2D.load_equal_chunks();
    for (size_t col = 0; col < data_2D.data().dims(1); ++col) {
        ASSERT_TRUE(af::allTrue<bool>(data_2D.data().col(col) == (float)this->rank_));
    }
}

TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_1D_INT_CUDA_TEST) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    juml::Backend::set(juml::Backend::CUDA);
    juml::Dataset data_1D(FILE_PATH, ONE_D_INT);

    data_1D.load_equal_chunks();
    for (size_t col = 0; col < data_1D.data().dims(1); ++col) {
        ASSERT_TRUE(af::allTrue<bool>(data_1D.data().col(col) == this->rank_));
    }
}

TEST_F(DATASET_TEST, LOAD_EQUAL_CHUNKS_2D_INT_CUDA_TEST) {
    juml::Backend::set(juml::Backend::CUDA);
    juml::Dataset data_2D(FILE_PATH, TWO_D_INT);

    data_2D.load_equal_chunks();

    for (size_t col = 0; col < data_2D.data().dims(1); ++col) {
        ASSERT_TRUE(af::allTrue<bool>(data_2D.data().col(col) == this->rank_));
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
