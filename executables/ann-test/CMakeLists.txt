ADD_EXECUTABLE(juml-ann-test ann-test.cpp)
TARGET_LINK_LIBRARIES(juml-ann-test classification ${CMAKE_THREAD_LIBS_INIT})

