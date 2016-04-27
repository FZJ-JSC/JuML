#include <arrayfire.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "core/Test.h"
#include "core/MPI.h"

static const int DIM_0 = 2;
static const int DIM_1 = 3;
static const int DIM_2 = 4;

#define GAUSSIAN_SUM(n) (((n) * (n) + (n)) / 2)

TEST_ALL(MPI_TEST, ALLGATHER_1D) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    af::array data = af::constant(rank, DIM_0);
    juml::mpi::allgather(data, MPI_COMM_WORLD);

    ASSERT_EQ(data.dims(0), DIM_0 * size);

    for (int i = 0; i < size; ++i) {
        ASSERT_TRUE(af::allTrue<bool>(data(af::seq(i * DIM_0, (i + 1) * DIM_0 - 1)) == i));
    }
}

TEST_ALL(MPI_TEST, ALLGATHER_2D) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    af::array data = af::constant(rank, DIM_0, DIM_1);
    juml::mpi::allgather(data, MPI_COMM_WORLD);

    ASSERT_EQ(data.dims(0), DIM_0);
    ASSERT_EQ(data.dims(1), DIM_1 * size);

    for (int i = 0; i < size; ++i) {
        ASSERT_TRUE(af::allTrue<bool>(data(af::span, af::seq(i * DIM_1, (i + 1) * DIM_1 - 1)) == i));
    }
}

TEST_ALL(MPI_TEST, ALLGATHER_3D) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    af::array data = af::constant(rank, DIM_0, DIM_1, DIM_2);
    juml::mpi::allgather(data, MPI_COMM_WORLD);

    ASSERT_EQ(data.dims(0), DIM_0);
    ASSERT_EQ(data.dims(1), DIM_1);
    ASSERT_EQ(data.dims(2), DIM_2 * size);

    for (int i = 0; i < size; ++i) {
        ASSERT_TRUE(af::allTrue<bool>(data(af::span, af::span, af::seq(i * DIM_2, (i + 1) * DIM_2 - 1)) == i));
    }
}

TEST_ALL(MPI_TEST, ALLGATHERV_1D) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    af::array data = af::constant(rank, rank + 1);
    juml::mpi::allgatherv(data, MPI_COMM_WORLD);

    ASSERT_EQ(data.dims(0), GAUSSIAN_SUM(size));

    for (int i = 0; i < size; ++i) {
        int start = GAUSSIAN_SUM(i);
        ASSERT_TRUE(af::allTrue<bool>(data(af::seq(start, start + i)) == i));
    }
}

TEST_ALL(MPI_TEST, ALLGATHERV_2D) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    af::array data = af::constant(rank, DIM_0, rank + 1);
    juml::mpi::allgatherv(data, MPI_COMM_WORLD);

    ASSERT_EQ(data.dims(0), DIM_0);
    ASSERT_EQ(data.dims(1), GAUSSIAN_SUM(size));

    for (int i = 0; i < size; ++i) {
        int start = GAUSSIAN_SUM(i);
        ASSERT_TRUE(af::allTrue<bool>(data(af::span, af::seq(start, start + i)) == i));
    }
}

TEST_ALL(MPI_TEST, ALLGATHERV_3D) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    af::array data = af::constant(rank, DIM_0, DIM_1, rank + 1);
    juml::mpi::allgatherv(data, MPI_COMM_WORLD);

    ASSERT_EQ(data.dims(0), DIM_0);
    ASSERT_EQ(data.dims(1), DIM_1);
    ASSERT_EQ(data.dims(2), GAUSSIAN_SUM(size));

    for (int i = 0; i < size; ++i) {
        int start = GAUSSIAN_SUM(i);
        ASSERT_TRUE(af::allTrue<bool>(data(af::span, af::span, af::seq(start, start + i)) == i));
    }
}

TEST_ALL(MPI_TEST, ALLREDUCE_INPLACE_3D) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    af::array min = af::constant(rank, DIM_0, DIM_1, DIM_2);
    af::array max = af::constant(rank, DIM_0, DIM_1, DIM_2);

    juml::mpi::allreduce_inplace(min, MPI_MIN, MPI_COMM_WORLD);
    juml::mpi::allreduce_inplace(max, MPI_MAX, MPI_COMM_WORLD);

    ASSERT_EQ(min.dims(0), DIM_0);
    ASSERT_EQ(min.dims(1), DIM_1);
    ASSERT_EQ(min.dims(2), DIM_2);
    ASSERT_EQ(max.dims(0), DIM_0);
    ASSERT_EQ(max.dims(1), DIM_1);
    ASSERT_EQ(max.dims(2), DIM_2);

    ASSERT_TRUE(af::allTrue<bool>(min == 0));
    ASSERT_TRUE(af::allTrue<bool>(max == size - 1));
}

TEST_ALL(MPI_TEST, EXSCAN_INPLACE_2D) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    af::array value = af::constant(rank, DIM_0, DIM_1);
    juml::mpi::exscan_inplace(value, MPI_SUM, MPI_COMM_WORLD);

    ASSERT_EQ(value.dims(0), DIM_0);
    ASSERT_EQ(value.dims(1), DIM_1);

    if (rank == 0)
        return;
    ASSERT_TRUE(af::allTrue<bool>(value == GAUSSIAN_SUM(rank - 1)));
}

TEST_ALL(MPI_TEST, SCAN_INPLACE_1D) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    af::array value = af::constant(rank, 1);
    juml::mpi::scan_inplace(value, MPI_SUM, MPI_COMM_WORLD);

    ASSERT_EQ(value.dims(0), 1);
    ASSERT_TRUE(af::allTrue<bool>(value == GAUSSIAN_SUM(rank)));
}

int main(int argc, char** argv) {
    int result = -1;
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ::testing::InitGoogleTest(&argc, argv);

    // suppress the output from the other ranks
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
