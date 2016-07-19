CUDA_ADD_EXECUTABLE(juml-ann-train-classifier ann-train-classifier.cpp)
TARGET_LINK_LIBRARIES(juml-ann-train-classifier classification ${CMAKE_THREAD_LIBS_INIT})

