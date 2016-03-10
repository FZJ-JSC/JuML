/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: MPI.cpp
*
* Description: Implementation of the MPI convenience wrappers
*
* Maintainer: m.goetz
*
* Email: murxman@gmail.com
*/

#include <stdexcept>

#include "core/MPI.h"

namespace juml {
namespace mpi {
    bool can_use_device_pointer(const af::array& data) {
        return af::getBackendId(data) == AF_BACKEND_CPU;
    }

    MPI_Datatype get_MPI_type(const af::array& data) {
        switch (data.type()) {
            case b8:
                return MPI_CHAR;
            case u8:
                return MPI_UNSIGNED_CHAR;
            case s16:
                return MPI_SHORT;
            case u16:
                return MPI_UNSIGNED_SHORT;
            case s32:
                return MPI_INT;
            case u32:
                return MPI_UNSIGNED;
            case s64:
                return MPI_LONG_LONG;
            case u64:
                return MPI_UNSIGNED_LONG_LONG;
            case f32:
                return MPI_FLOAT;
            case f64:
                return MPI_DOUBLE;
            case c32:
                return MPI_COMPLEX;
            case c64:
                return MPI_DOUBLE_COMPLEX;
            default:
                throw std::domain_error("Could not convert datatype to MPI equivalent");
        }
    }

    void allreduce_inplace(af::array& data, MPI_Op op, MPI_Comm comm) {
        data.eval(); // safeguard that af op tree is committed, used to be bugged as of af 2.14 (could not get dev ptr)
        void* data_pointer = nullptr;
        bool  use_device_pointer = can_use_device_pointer(data);

        if (use_device_pointer) {
            data_pointer = reinterpret_cast<void*>(data.device<unsigned char>());
        } else {
            data_pointer = reinterpret_cast<void*>(new unsigned char[data.bytes()]);
            data.host(data_pointer);
        }
        MPI_Allreduce(MPI_IN_PLACE, data_pointer, data.elements(), get_MPI_type(data), op, comm);
        if (use_device_pointer) {
            data.unlock();
        } else {
            af_write_array(data.get(), data_pointer, data.bytes(), afHost);
            delete[] data_pointer;
        }
    }
} // namespace mpi
} // namespace juml
