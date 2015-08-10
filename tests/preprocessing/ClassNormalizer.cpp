#include <exception>
#include <gtest/gtest.h>
#include <iostream>
#include <mpi.h>
#include <string>

#include "data/Dataset.h"
#include "preprocessing/ClassNormalizer.h"

static const std::string FILE_PATH = "../../../datasets/random_class_labels.h5";
static const std::string DATA_SET = "labels";
 
TEST (CLASS_NORMALIZER_TEST, MAPPING_TEST) {
    juml::Dataset<int> labels(FILE_PATH, DATA_SET);
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
        ASSERT_EQ(class_normalizer.invert(transformed), original);
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
