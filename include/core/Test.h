/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: Test.h
*
* Description: Collection of helper macros and function for unit-testing.
*
* Maintainer: m.goetz
*
* Email: murxman@gmail.com
*/

#ifndef JUML_TEST_H
#define JUML_TEST_H

#include <arrayfire.h>

#include "core/Backend.h"

#define TEST_FUNCTION_NAME(GROUP, NAME) test_multiple_af_backends_ ## GROUP ## _ ## NAME
#define TEST_FUNCTION_BODY(GROUP, NAME, PLATFORM) \
    juml::Backend::set(PLATFORM); \
    TEST_FUNCTION_NAME(GROUP, NAME)(PLATFORM);

#define TEST_INTERCEPTOR_CLASS_NAME(GROUP, NAME) GROUP##_##NAME##_Test_Intercept
#define TEST_INTERCEPTOR_METHOD_NAME test_multiple_af_backends
#define TEST_INTERCEPTOR_CLASS_DEFINITION(GROUP, NAME) \
class TEST_INTERCEPTOR_CLASS_NAME(GROUP, NAME) : public GROUP { \
public: \
    TEST_INTERCEPTOR_CLASS_NAME(GROUP, NAME)() {} \
protected: \
    void TEST_INTERCEPTOR_METHOD_NAME(int BACKEND); \
};
#define TEST_INTERCEPTOR_BODY(PLATFORM) \
    juml::Backend::set(PLATFORM); \
    this->TEST_INTERCEPTOR_METHOD_NAME(PLATFORM);

#define TEST_ALL_CPU(GROUP, NAME) \
TEST(GROUP, NAME ## _CPU) { TEST_FUNCTION_BODY(GROUP, NAME, juml::Backend::CPU); }
#define TEST_ALL_CPU_F(GROUP, NAME) \
GTEST_TEST_(GROUP, NAME ## _CPU, TEST_INTERCEPTOR_CLASS_NAME(GROUP, NAME), \
    ::testing::internal::GetTypeId<GROUP>()) { TEST_INTERCEPTOR_BODY(juml::Backend::CPU); }

#ifdef JUML_CUDA
#define TEST_ALL_CUDA(GROUP, NAME) \
TEST(GROUP, NAME ## _CUDA) { TEST_FUNCTION_BODY(GROUP, NAME, juml::Backend::CUDA); }
#define TEST_ALL_CUDA_F(GROUP, NAME) \
GTEST_TEST_(GROUP, NAME ## _CUDA, TEST_INTERCEPTOR_CLASS_NAME(GROUP, NAME), \
    ::testing::internal::GetTypeId<GROUP>()) { TEST_INTERCEPTOR_BODY(juml::Backend::CUDA); }
#else
#define TEST_ALL_CUDA(GROUP, NAME)
#define TEST_ALL_CUDA_F(GROUP, NAME)
#endif // JUML_CUDA

#ifdef JUML_OPENCL
#define TEST_ALL_OPENCL(GROUP, NAME) \
TEST(GROUP, NAME ## _OPENCL) { TEST_FUNCTION_BODY(GROUP, NAME, juml::Backend::OPENCL); }
#define TEST_ALL_OPENCL_F(GROUP, NAME) \
GTEST_TEST_(GROUP, NAME ## _OPENCL, TEST_INTERCEPTOR_CLASS_NAME(GROUP, NAME), \
    ::testing::internal::GetTypeId<GROUP>()) { TEST_INTERCEPTOR_BODY(juml::Backend::OPENCL); }
#else
#define TEST_ALL_OPENCL(GROUP, NAME)
#define TEST_ALL_OPENCL_F(GROUP, NAME)
#endif // JUML_OPENCL

#define TEST_ALL(GROUP, NAME) \
void TEST_FUNCTION_NAME(GROUP, NAME)(int BACKEND); \
TEST_ALL_CPU(GROUP, NAME) \
TEST_ALL_CUDA(GROUP, NAME) \
TEST_ALL_OPENCL(GROUP, NAME) \
void TEST_FUNCTION_NAME(GROUP, NAME)(int BACKEND)

#define TEST_ALL_F(GROUP, NAME) \
TEST_INTERCEPTOR_CLASS_DEFINITION(GROUP, NAME) \
TEST_ALL_CPU_F(GROUP, NAME) \
TEST_ALL_CUDA_F(GROUP, NAME) \
TEST_ALL_OPENCL_F(GROUP, NAME) \
void TEST_INTERCEPTOR_CLASS_NAME(GROUP, NAME)::TEST_INTERCEPTOR_METHOD_NAME(int Backend)

#endif //JUML_TEST_H
