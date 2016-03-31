/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: Definitions.h
*
* Description: Collection of datatypes and constants for JuML
*
* Maintainer: m.goetz
*
* Email: murxman@gmail.com
*/

#ifndef BACKEND_H
#define BACKEND_H

#include <arrayfire.h>

namespace juml {
    class Backend {
    protected:
        int backend_;
    
    public:
        static const int CPU    = AF_BACKEND_CPU;
        static const int CUDA   = AF_BACKEND_CUDA;
        static const int OPENCL = AF_BACKEND_OPENCL;

        static void set(int backend);
        static int of(const af::array& data);
        
        Backend(int backend);
        int get() const;
    };
} // namespace juml

#endif // BACKEND_H

