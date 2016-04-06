#include <exception>
#include <gtest/gtest.h>
#include <iostream>
#include <mpi.h>
#include <string>

#include "core/Test.h"
#include "data/Dataset.h"
#include "preprocessing/ClassNormalizer.h"

static const std::string FILE_PATH = JUML_DATASETS"/random_class_labels.h5";
static const std::string DATA_SET = "labels";

TEST_ALL(CLASS_NORMALIZER_TEST, MAPPING_TEST) {
    juml::Dataset labels(FILE_PATH, DATA_SET);
    labels.load_equal_chunks();
    juml::ClassNormalizer class_normalizer;
    class_normalizer.index(labels);

    // check number of classes
    ASSERT_EQ(class_normalizer.n_classes(), 19);

    // check transform and invert
    // skip class label -5 because it is missing
    int original, transformed;
    for (original = -10, transformed = 0; original < 10; ++original, ++transformed) {
        if (original == -5) {
            --transformed;
            continue;
        }
        ASSERT_EQ(class_normalizer.transform(original), transformed);
        ASSERT_EQ(class_normalizer.invert<int>(transformed), original);
    }
}

TEST_ALL(CLASS_NORMALIZER_TEST, MAPPING_EXCEPTION) {
    juml::Dataset labels(FILE_PATH, DATA_SET);
    labels.load_equal_chunks();
    juml::ClassNormalizer class_normalizer;

    // test non row-vector inputs
    ASSERT_THROW(class_normalizer.index(juml::Dataset(af::constant(0, 100, 1, 1, 1))), std::invalid_argument);
    ASSERT_THROW(class_normalizer.index(juml::Dataset(af::constant(0, 1, 1, 100, 1))), std::invalid_argument);
    ASSERT_THROW(class_normalizer.index(juml::Dataset(af::constant(0, 1, 1, 1, 100))), std::invalid_argument);

    // set up a working normalizer and wrong labels
    class_normalizer.index(labels);
    int out_of_range_transform = af::max<int>(class_normalizer.classes()) + 1;
    int out_of_range_invert = static_cast<int>(class_normalizer.n_classes()) + 1;

    // singular input
    ASSERT_THROW(class_normalizer.transform<int>(out_of_range_transform), std::invalid_argument);
    ASSERT_THROW(class_normalizer.invert<int>(out_of_range_invert), std::invalid_argument);
}

TEST_ALL(CLASS_NORMALIZER_TEST, VECTOR_MAPPING) {
    juml::Dataset labels(FILE_PATH, DATA_SET);
    labels.load_equal_chunks();
    juml::ClassNormalizer class_normalizer;
    class_normalizer.index(labels);

    af::array transformed = class_normalizer.transform(labels.data());
    ASSERT_TRUE(af::allTrue<bool>(class_normalizer.invert(transformed) == labels.data()));
}

TEST_ALL(CLASS_NORMALIZER_TEST, VECTOR_MAPPING_EXCEPTION) {
    juml::Dataset labels(FILE_PATH, DATA_SET);
    labels.load_equal_chunks();

    juml::ClassNormalizer class_normalizer;
    class_normalizer.index(labels);

    // test misshapes vectors
    ASSERT_THROW(class_normalizer.transform(af::constant(-1, 100, 1, 1, 1)), std::invalid_argument);
    ASSERT_THROW(class_normalizer.transform(af::constant(-1, 1, 1, 100, 1)), std::invalid_argument);
    ASSERT_THROW(class_normalizer.transform(af::constant(-1, 1, 1, 1, 100)), std::invalid_argument);

    ASSERT_THROW(class_normalizer.invert(af::constant(-1, 100, 1, 1, 1)), std::invalid_argument);
    ASSERT_THROW(class_normalizer.invert(af::constant(-1, 1, 1, 100, 1)), std::invalid_argument);
    ASSERT_THROW(class_normalizer.invert(af::constant(-1, 1, 1, 1, 100)), std::invalid_argument);

    // test faulty input
//    ASSERT_THROW(class_normalizer.transform(af::constant(100, 1, 100)), std::invalid_argument);
    ASSERT_THROW(class_normalizer.invert(af::constant(100, 1, 100)), std::invalid_argument);
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
