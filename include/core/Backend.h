/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: Backend.h
*
* Description: Backend header
*
* Maintainer: m.goetz
*
* Email: murxman@gmail.com
*/

#ifndef JUML_CORE_BACKEND_H_
#define JUML_CORE_BACKEND_H_

#include <arrayfire.h>

namespace juml {
    class Backend {
     protected:
        int backend_;

     public:
        /**
         * @var   CPU
         * @brief Symbolic variable for the CPU backend
         */
        static const int CPU = AF_BACKEND_CPU;
        /**
         * @var   CUDA
         * @brief Symbolic variable for the CUDA backend
         */
        static const int CUDA = AF_BACKEND_CUDA;
        /**
         * @var   OPENCL
         * @brief Symbolic variable for the OpenCL backend
         */
        static const int OPENCL = AF_BACKEND_OPENCL;

        /**
         * set
         *
         * Allows to set the current processing backend, on of Backend::CPU, Backend::CUDA or Backend::OPENCL.
         *
         * @param backend - The processing backend
         */
        static void set(int backend);

        /**
         * of
         *
         * Inquires the storage backend of an arrayfire array.
         *
         * @param data - The data array
         * @returns The processing backend
         */
        static int of(const af::array& data);

        /**
         * Backend constructor
         *
         * Constructs a new backend object that automatically sets current processing backend. Is used in algorithms to
         * intercept any subsequent arrayfire array constructions so that they will be created on the proper backend.
         *
         * @param backend - The processing backend.
         */
        explicit Backend(int backend);

        /**
         * get
         *
         * @returns The processing backend of a Backend object.
         */
        int get() const;
    };
} // namespace juml

#endif  // JUML_CORE_BACKEND_H_

