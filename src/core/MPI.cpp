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

#include "core/Backend.h"
#include "core/MPI.h"

namespace juml {
namespace mpi {
    bool can_use_device_pointer(const af::array& data) {
        return Backend::of(data) == Backend::CPU;
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

    int allgather(af::array& data, MPI_Comm comm, dim_t merge) {
        // obtain the read buffer
        data.eval();
        void* data_pointer = nullptr;
        bool  use_device_pointer = can_use_device_pointer(data);

        if (use_device_pointer) {
            data_pointer = reinterpret_cast<void*>(data.device<unsigned char>());
        } else {
            data_pointer = reinterpret_cast<void*>(new unsigned char[data.bytes()]);
            data.host(data_pointer);
        }

        // allocate the gather buffer
        int mpi_size;
        MPI_Comm_size(comm, &mpi_size);
        void* gather_buffer = new unsigned char[data.bytes() * mpi_size];

        // exchange the message
        MPI_Datatype type = get_MPI_type(data);
        int error = MPI_Allgather(data_pointer, (int)data.elements(), type,
                                  gather_buffer, (int)data.elements(), type, comm);
        if (use_device_pointer) {
            data.unlock();
        } else {
            delete[] reinterpret_cast<unsigned char *>(data_pointer);
        }
        if (error != MPI_SUCCESS) {
            delete[] reinterpret_cast<unsigned char*>(gather_buffer);
            return error;
        }

        // copy the data back
        af::dim4 dimensions = data.dims();
        dimensions[merge] *= mpi_size;
        data = af::constant(0, dimensions, data.type());
        af_write_array(data.get(), gather_buffer, data.bytes(), afHost);
        delete[] reinterpret_cast<unsigned char*>(gather_buffer);

        return MPI_SUCCESS;
    }

    int allgatherv(af::array& data, MPI_Comm comm, dim_t merge) {
        // mpi book-keeping and sanity check for number of counts
        int mpi_rank;
        int mpi_size;

        MPI_Comm_rank(comm, &mpi_rank);
        MPI_Comm_size(comm, &mpi_size);

        // exchange the element counts and displacements
        int counts[mpi_size];
        int displacements[mpi_size];
        int total_elements = 0;
        counts[mpi_rank] = static_cast<int>(data.elements());
        displacements[0] = 0;

        MPI_Allgather(counts + mpi_rank, 1, MPI_INT, counts, 1, MPI_INT, comm);
        for (int i = 1; i < mpi_size; ++i) {
            displacements[i] = displacements[i - 1] + counts[i - 1];
            total_elements += counts[i];
        }
        total_elements += counts[0];

        // prepare the source and target buffers
        bool use_device_pointer = can_use_device_pointer(data);
        void* gather_buffer = new unsigned char[data.bytes() / data.elements() * total_elements];
        void* data_buffer = nullptr;

        data.eval();
        if (use_device_pointer) {
            data_buffer = reinterpret_cast<void*>(data.device<unsigned char>());
        } else {
            data_buffer = reinterpret_cast<void*>(new unsigned char[data.bytes()]);
            data.host(data_buffer);
        }

        // exchange the actual data
        MPI_Datatype type = get_MPI_type(data);
        int error = MPI_Allgatherv(data_buffer, counts[mpi_rank], type, gather_buffer, counts, displacements, type, comm);

        // release resources
        if (use_device_pointer) {
            data.unlock();
        } else {
            delete[] reinterpret_cast<unsigned char *>(data_buffer);
        }
        if (error != MPI_SUCCESS) {
            delete[] reinterpret_cast<unsigned char*>(gather_buffer);
            return error;
        }

        // copy the data back into the target array
        af::dim4 dimensions = data.dims();
        dimensions[merge] = total_elements / (data.elements() / data.dims(static_cast<unsigned int>(merge)));
        data = af::constant(0, dimensions, data.type());
        af_write_array(data.get(), gather_buffer, data.bytes() / data.elements() * total_elements, afHost);
        delete[] reinterpret_cast<unsigned char*>(gather_buffer);

        return MPI_SUCCESS;
    }

    int allreduce_inplace(af::array& data, MPI_Op op, MPI_Comm comm) {
        return inplace_reduction_collective(data, MPI_Allreduce, op, comm);
    }

    int exscan_inplace(af::array& data, MPI_Op op, MPI_Comm comm) {
        return inplace_reduction_collective(data, MPI_Exscan, op, comm);
    }

    int scan_inplace(af::array& data, MPI_Op op, MPI_Comm comm) {
        return inplace_reduction_collective(data, MPI_Scan, op, comm);
    }

    int inplace_reduction_collective(af::array &data, ReductionCollective function, MPI_Op op, MPI_Comm comm) {
        data.eval(); // safeguard that af op tree is committed, used to be bugged as of af 2.14 (could not get dev ptr)
        void* data_pointer = nullptr;
        bool  use_device_pointer = can_use_device_pointer(data);

        if (use_device_pointer) {
            data_pointer = reinterpret_cast<void*>(data.device<unsigned char>());
        } else {
            data_pointer = reinterpret_cast<void*>(new unsigned char[data.bytes()]);
            data.host(data_pointer);
        }

        int error = function(MPI_IN_PLACE, data_pointer, (int)data.elements(), get_MPI_type(data), op, comm);
        if (use_device_pointer) {
            data.unlock();
        } else {
            af_write_array(data.get(), data_pointer, data.bytes(), afHost);
            delete[] reinterpret_cast<unsigned char*>(data_pointer);
        }

        return error;
    }
} // namespace mpi
} // namespace juml
