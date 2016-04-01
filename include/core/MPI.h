/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: MPI.h
*
* Description: Convenience wrappers that allow to communicate af::arrays over the wire
*
* Maintainer: m.goetz
*
* Email: murxman@gmail.com
*/

#ifndef JUML_MPI_H
#define JUML_MPI_H

#include <arrayfire.h>
#include <mpi.h>

namespace juml {
namespace mpi {
    typedef int (*ReductionCollective)(const void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm);

    bool can_use_device_pointer(const af::array& data);
    MPI_Datatype get_MPI_type(const af::array& data);

    int allgather(af::array& data, MPI_Comm comm, dim_t merge=1);
    int allgatherv(af::array& data, MPI_Comm comm, dim_t merge=1);

    int allreduce_inplace(af::array& data, MPI_Op op, MPI_Comm comm);
    int exscan_inplace(af::array& data, MPI_Op op, MPI_Comm comm);
    int scan_inplace(af::array& data, MPI_Op op, MPI_Comm comm);
    int inplace_reduction_collective(af::array& data, ReductionCollective function, MPI_Op op, MPI_Comm comm);

} // namespace mpi
} // namespace juml

#endif //JUML_MPI_H
