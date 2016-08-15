IF(CUDA_FOUND)
    CUDA_ADD_EXECUTABLE(juml-ann-train-classifier ann-train-classifier.cpp)
ELSE()
    ADD_EXECUTABLE(juml-ann-train-classifier ann-train-classifier.cpp)
    MESSAGE(STATUS "CUDA-Aware MPI will not work for juml-ann-train-classifier, because CUDA was not found.")
ENDIF()
TARGET_LINK_LIBRARIES(juml-ann-train-classifier classification ${CMAKE_THREAD_LIBS_INIT})

