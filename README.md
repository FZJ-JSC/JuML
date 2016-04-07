# JuML

JuML (Juelich Machine Learning library) is a parallel, high-performance software library that makes large-scale data
analysis in common HPC-setups simple. It supports common machine learning algorithms such Gaussian Naive Bayes, K-Means
or Neural Networks, bundled with a rich set of convenience functions such as data normalization. JuML provides high-
level APIs for C++ and Python that allows to compute both, on native CPUs as well as accelerators like GPGPUs.

## Dependencies

JuML requires a number of mandatory software packages, and has additional dependencies. The build process, described
below, will automatically check for their presence, but will not install them. Make sure you have everything set up in
advance:

Mandatory:

* cmake
* arrayfire
* mpi
* hdf5

Optional:

* cuda
* opencl
* doxygen
* python
* numpy
* swig

## Building JuML

JuML is built using [CMake](https://cmake.org/). To build it, run the following in JuML's main directory:

    `mkdir build && cd build && cmake .. && make`

## Documentation

The code is documented in the source headers in [Doxygen](http://www.stack.nl/~dimitri/doxygen/) format. A browsable
HTML version can be generated with doxygen:

    `doxygen <CONFIG_FILE>`

Further information can be obtained [here](http://www.stack.nl/~dimitri/doxygen/manual/starting.html).

## Test suite

Testing requires that JuML is built (see above). Once the library is built, the test suite can be executed:

    `make vtest`

Individual tests for modules can be run using:

    `ctest -R <TEST_NAME>`

For more in-depth information about CMake's testing capabilities, refer to [CMake Testing](https://cmake.org/Wiki/CMake/Testing_With_CTest#Running_Individual_Tests).

## License terms

JuML is published under the liberal terms of the BSD License. Although the BSD License does not require you to share
any modifications you make to the source code, you are very much encouraged and invited to contribute back your
modifications to the community, preferably in a Github fork, of course.
