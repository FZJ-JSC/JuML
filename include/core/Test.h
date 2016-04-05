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

#define TEST_ALL_CPU(GROUP, NAME)   TEST(GROUP, NAME ## _CPU)   { test_multiple_af_backends_ ## GROUP ## _ ## NAME(juml::Backend::CPU); }
#define TEST_ALL_CPU_F(GROUP, NAME) TEST_F(GROUP, NAME ## _CPU) { test_multiple_af_backends_ ## GROUP ## _ ## NAME(juml::Backend::CPU); }

#ifdef JUML_CUDA
#define TEST_ALL_CUDA(GROUP, NAME)   TEST(GROUP, NAME ## _CUDA)   { test_multiple_af_backends_ ## GROUP ## _ ## NAME(juml::Backend::CUDA); }
#define TEST_ALL_CUDA_F(GROUP, NAME) TEST_F(GROUP, NAME ## _CUDA) { test_multiple_af_backends_ ## GROUP ## _ ## NAME(juml::Backend::CUDA); }
#else
#define TEST_ALL_CUDA(GROUP, NAME)
#define TEST_ALL_CUDA_F(GROUP, NAME)
#endif // JUML_CUDA

#ifdef JUML_OPENCL
#define TEST_ALL_OPENCL(GROUP, NAME)   TEST(GROUP, NAME ## _OPENCL)   { test_multiple_af_backends_ ## GROUP ## _ ## NAME(juml::Backend::OPENCL); }
#define TEST_ALL_OPENCL_F(GROUP, NAME) TEST_F(GROUP, NAME ## _OPENCL) { test_multiple_af_backends_ ## GROUP ## _ ## NAME(juml::Backend::OPENCL); }
#else
#define TEST_ALL_OPENCL(GROUP, NAME)
#define TEST_ALL_OPENCL_F(GROUP, NAME)
#endif // JUML_OPENCL

#define TEST_ALL(GROUP, NAME) void test_multiple_af_backends_ ## GROUP ## _ ## NAME(int BACKEND); \
TEST_ALL_CPU(GROUP, NAME) \
TEST_ALL_CUDA(GROUP, NAME) \
TEST_ALL_OPENCL(GROUP, NAME) \
void test_multiple_af_backends_ ## GROUP ## _ ## NAME(int BACKEND)

#define TEST_ALL_F(GROUP, NAME) void test_multiple_af_backends_ ## GROUP ## _ ## NAME(int BACKEND); \
TEST_ALL_CPU_F(GROUP, NAME) \
TEST_ALL_CUDA_F(GROUP, NAME) \
TEST_ALL_OPENCL_F(GROUP, NAME) \
void test_multiple_af_backends_ ## GROUP ## _ ## NAME(int BACKEND)

#endif //JUML_TEST_H
