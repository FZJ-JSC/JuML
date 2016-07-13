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

#include <math.h>
#include <stdexcept>

#include "core/Backend.h"
#include "core/MPI.h"
#include "mpi-ext.h"

namespace juml {
namespace mpi {
    bool cuda_aware_mpi_available = false;

    void init(int *argc, char ***argv) {
        MPI_Init(argc, argv);
#if defined(MPIX_CUDA_AWARE_SUPPORT)
        cuda_aware_mpi_available = 1 == MPIX_Query_cuda_support();
#else
	cuda_aware_mpi_available = false;
#endif
    }

    bool can_use_device_pointer(const af::array& data) {
        return Backend::of(data) == Backend::CPU || (cuda_aware_mpi_available && Backend::of(data) == Backend::CUDA);
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

    int allgather(af::array& data, MPI_Comm comm) {
        // MPI administration
        int mpi_size;
        MPI_Comm_size(comm, &mpi_size);
        MPI_Datatype type = get_MPI_type(data);

        // allocate target memory
        af::dim4 dimensions = data.dims();
        dimensions[data.numdims() - 1] *= mpi_size;
        af::array target = af::array(dimensions, data.type());

        // force evaluations of operations and allocation
        data.eval();
        target.eval();

        // do the actual data exchange
        if (can_use_device_pointer(data)) {
            void* data_pointer = reinterpret_cast<void *>(data.device<unsigned char>());
            void* gather_buffer = reinterpret_cast<void *>(target.device<unsigned char>());
            int error = MPI_Allgather(data_pointer, (int)data.elements(), type, gather_buffer, (int)data.elements(), type, comm);
            data.unlock();
            target.unlock();
            if (error != MPI_SUCCESS) {
                return error;
            }
        } else {
            void* data_pointer = reinterpret_cast<void*>(new unsigned char[data.bytes()]);
            void* gather_buffer = new unsigned char[data.bytes() * mpi_size];
            data.host(data_pointer);
            int error = MPI_Allgather(data_pointer, (int)data.elements(), type, gather_buffer, (int)data.elements(), type, comm);
            if (error != MPI_SUCCESS) {
                delete[] reinterpret_cast<unsigned char *>(data_pointer);
                delete[] reinterpret_cast<unsigned char*>(gather_buffer);
                data.unlock();
                return error;
            }
            af_write_array(target.get(), gather_buffer, target.bytes(), afHost);
            delete[] reinterpret_cast<unsigned char *>(data_pointer);
            delete[] reinterpret_cast<unsigned char*>(gather_buffer);
            data.unlock();
        }
        data = target;

        return MPI_SUCCESS;
    }

    int allgatherv(af::array& data, MPI_Comm comm) {
        // mpi book-keeping
        int mpi_rank, mpi_size, mpi_error;
        MPI_Comm_rank(comm, &mpi_rank);
        MPI_Comm_size(comm, &mpi_size);

        // exchange the element counts and displacements
        int counts[mpi_size + 1] = {0};
        int displacements[mpi_size];
        int total_elements = 0;
        counts[mpi_rank] = static_cast<int>(data.elements());
        counts[mpi_size] = static_cast<int>(data.numdims());
        displacements[0] = 0;

        mpi_error = MPI_Allreduce(MPI_IN_PLACE, counts, mpi_size + 1, MPI_INT, MPI_SUM, comm);
        for (int i = 1; i < mpi_size; ++i) {
            displacements[i] = displacements[i - 1] + counts[i - 1];
            total_elements += counts[i];
        }
        int num_dims = (int)std::ceil(counts[mpi_size] / (float)mpi_size);
        total_elements += counts[0];

        // prepare the source and target buffers
        bool use_device_pointer = can_use_device_pointer(data);
        MPI_Datatype type = get_MPI_type(data);

        // allocate target
        af::dim4 dimensions = data.dims();
        dimensions[num_dims - 1] = total_elements / (data.elements() / data.dims(static_cast<unsigned int>(num_dims - 1)));
        af::array target = af::array(dimensions, data.type());
        data.eval();
        target.eval();

        if (use_device_pointer) {
            void* data_buffer = reinterpret_cast<void*>(data.device<unsigned char>());
            void* gather_buffer = reinterpret_cast<void*>(target.device<unsigned char>());
            mpi_error = MPI_Allgatherv(data_buffer, counts[mpi_rank], type, gather_buffer, counts, displacements, type, comm);
            data.unlock();
            target.unlock();
        } else {
            void* data_buffer = reinterpret_cast<void*>(new unsigned char[data.bytes()]);
            void* gather_buffer = new unsigned char[data.bytes() / data.elements() * total_elements];
            data.host(data_buffer);
            mpi_error = MPI_Allgatherv(data_buffer, counts[mpi_rank], type, gather_buffer, counts, displacements, type, comm);
            if (mpi_error != MPI_SUCCESS) {
                delete[] reinterpret_cast<unsigned char*>(data_buffer);
                delete[] reinterpret_cast<unsigned char*>(gather_buffer);
                return mpi_error;
            }
            af_write_array(target.get(), gather_buffer, data.bytes() / data.elements() * total_elements, afHost);
            data.unlock();
            delete[] reinterpret_cast<unsigned char*>(data_buffer);
            delete[] reinterpret_cast<unsigned char*>(gather_buffer);
        }
        data = target;

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
