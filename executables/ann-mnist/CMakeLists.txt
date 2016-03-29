ADD_EXECUTABLE(juml-ann-mnist ann-mnist.cpp)
TARGET_LINK_LIBRARIES(juml-ann-mnist classification ${CMAKE_THREAD_LIBS_INIT})

