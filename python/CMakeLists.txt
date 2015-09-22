FIND_PACKAGE(SWIG)
IF(SWIG_FOUND)
    INCLUDE(${SWIG_USE_FILE})

    FIND_PACKAGE(PythonLibs REQUIRED)
    FIND_PACKAGE(PythonInterp REQUIRED)
    INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})

    MESSAGE(STATUS "Python libs: " ${PYTHON_LIBRARIES}) 
    
    EXECUTE_PROCESS(COMMAND
        ${PYTHON_EXECUTABLE} -c "import mpi4py"
        OUTPUT_VARIABLE Mpi4Py_FOUND
        ERROR_VARIABLE  Mpi4Py_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    IF (NOT ${Mpi4Py_ERROR} EQUAL 0)
        MESSAGE(FATAL_ERROR "mpi4py not found: " ${Mpi4Py_ERROR}) 
    ELSE ()
        EXECUTE_PROCESS(COMMAND
            ${PYTHON_EXECUTABLE} -c "import mpi4py; print(mpi4py.get_include())"
            OUTPUT_VARIABLE Mpi4Py_INCLUDE_PATH
            ERROR_VARIABLE  Mpi4Py_INCLUDE_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        EXECUTE_PROCESS(COMMAND
            ${PYTHON_EXECUTABLE} -c "import mpi4py; print(mpi4py.__version__)"
            OUTPUT_VARIABLE Mpi4Py_VERSION
            ERROR_VARIABLE  Mpi4Py_VERSION_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        SET(Mpi4Py_REQUIRED_VERSION "1.3")

        INCLUDE_DIRECTORIES(${Mpi4Py_INCLUDE_PATH})
        MESSAGE(STATUS "Found mpi4py: " ${Mpi4Py_INCLUDE_PATH} " (found version " ${Mpi4Py_VERSION} ")") 

        IF(${Mpi4Py_VERSION} VERSION_LESS ${Mpi4Py_REQUIRED_VERSION})
            MESSAGE(FATAL_ERROR "mpi4py wrong version: " ${Mpi4Py_VERSION} " < " ${Mpi4Py_REQUIRED_VERSION}) 
        ENDIF()
    ENDIF ()   

    EXECUTE_PROCESS(COMMAND
        ${PYTHON_EXECUTABLE} -c "import numpy"
        OUTPUT_VARIABLE Numpy_FOUND
        ERROR_VARIABLE  Numpy_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    IF (NOT ${Numpy_ERROR} EQUAL 0)

        MESSAGE(FATAL_ERROR "numpy not found: " ${Numpy_ERROR}) 
    ELSE ()
        EXECUTE_PROCESS(COMMAND
            ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
            OUTPUT_VARIABLE Numpy_INCLUDE_PATH
            ERROR_VARIABLE  Numpy_INCLUDE_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        EXECUTE_PROCESS(COMMAND
            ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.__version__)"
            OUTPUT_VARIABLE Numpy_VERSION
            ERROR_VARIABLE  Numpy_VERSION_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        MESSAGE(STATUS "Found numpy: " ${Numpy_INCLUDE_PATH} " (found version " ${Numpy_VERSION} ")") 
        SET(Numpy_REQUIRED_VERSION "1.9")
        IF(${Numpy_VERSION} VERSION_LESS ${Numpy_REQUIRED_VERSION})
            MESSAGE(FATAL_ERROR "numpy wrong version: " ${Numpy_VERSION} " < " ${Numpy_REQUIRED_VERSION}) 
        ENDIF()
    ENDIF ()

    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

    SET(CMAKE_SWIG_FLAGS "")

    SET_SOURCE_FILES_PROPERTIES(JuML.i PROPERTIES CPLUSPLUS ON)
    SET_SOURCE_FILES_PROPERTIES(JuML.i PROPERTIES SWIG_FLAGS  "-wall;-c++")

    FILE(GLOB JuML_HEADER ${JuML_INLUDE_DIRECTORIES}/*/*.h)
    SET(SWIG_MODULE_JuML_EXTRA_DEPS 
        ${JuML_HEADER}
        armanpy.hpp
    )

    SWIG_ADD_MODULE(JuML python JuML.i)
    SWIG_LINK_LIBRARIES(JuML ${PYTHON_LIBRARIES} preprocessing classification)
ENDIF(SWIG_FOUND)
