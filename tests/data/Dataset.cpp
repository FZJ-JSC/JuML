#include <armadillo>
#include <gtest/gtest.h>

#include "data/Dataset.h"

class TestableDataset : public juml::Dataset {
    public: using juml::Dataset::load_equal_chunks;
public:
    TestableDataset(const std::string& filename, const std::string& data_set, MPI_Comm comm)
        : juml::Dataset(filename, data_set, comm) {};

    TestableDataset(const std::string& filename, const std::string& data_set)
        : juml::Dataset(filename, data_set, MPI_COMM_WORLD) {};
};

TEST (DATASET_TEST, LOAD_EQUAL_CHUNKS_1D_TEST) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    TestableDataset data_1D = TestableDataset("../../datasets/mpi_ranks.h5", "1D");
    data_1D.load_equal_chunks();
    for (size_t row = 0; row < data_1D.data().n_elem; ++row) {
        ASSERT_FLOAT_EQ(data_1D.data()[row], (float)rank);
    }
}


TEST (DATASET_TEST, LOAD_EQUAL_CHUNKS_2D_TEST) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    TestableDataset data_2D = TestableDataset("datasets/mpi_ranks.h5", "2D");
    data_2D.load_equal_chunks();

    for (size_t row = 0; row < data_2D.data().n_elem; ++row) {
        ASSERT_FLOAT_EQ(data_2D.data()[row], (float)rank);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}
