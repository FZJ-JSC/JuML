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

    /**
     * can_use_device_pointer
     *
     * Checks whether the device pointer of an af array can be directly used in an MPI communication call or not. This
     * is for example the case for CPU arrays or setups that support CUDA-aware MPI.
     *
     * @param data - The arrayfire array to be checked
     * @returns True if device pointer can be directly used in MPI calls, false otherwise
     */
    bool can_use_device_pointer(const af::array& data);

    /**
     * get_MPI_type
     *
     * Finds the correct MPI type (e.g MPI_INT) corresponding to the type of an arrayfire array.
     *
     * @param   data - The arrayfire array to be checked
     * @returns The corresponding MPI type
     */
    MPI_Datatype get_MPI_type(const af::array& data);

    /**
     * allgather
     *
     * Performs an allgather operation using the passed data and on the given MPI communicator. Assumes that the data
     * portion sizes on all nodes match. The gathered data will be stored in a new arrayfire array and returned via the
     * input/output parameter data.
     *
     * @param data  - The input and output parameter for the gather data
     * @param comm  - The MPI communicator to perform the gather operation on
     * @param merge - The dimension on which to merge the data, defaults to 1 (sample dimension for 2D data)
     * @returns The MPI error code
     */
    int allgather(af::array& data, MPI_Comm comm, dim_t merge=1);
    /**
     * allgatherv
     *
     * Performs an allgatherv operation using the passed data and MPI communicator. Assumes that the data portions vary
     * in size along the merge dimension and inquires the displacements automatically. The gathered data will be
     * collected in a new arrayfire array and returned via the input/output parameter data.
     *
     * @param data  - The input and output parameter for the gather data
     * @param comm  - The MPI communicator to perform the gather operation on
     * @param merge - The dimension along which to merge the data, defaults to 1 (sample dimension for 2D data)
     * @returns The MPI error code
     */
    int allgatherv(af::array& data, MPI_Comm comm, dim_t merge=1);

    /**
     * allreduce_inplace
     *
     * Performs an allreduce operation using the passed data and MPI communicator. The input data will be overwritten
     * (inplace) during the reduction using operator op.
     *
     * @param data - The input and output parameter for the reduced data
     * @param op   - The MPI reduction operator handle (e.g. MPI_SUM)
     * @param comm - The MPI communicator to perform the reduction operation on
     * @returns The MPI error code
     */
    int allreduce_inplace(af::array& data, MPI_Op op, MPI_Comm comm);

    /**
     * exscan_inplace
     *
     * Performs an exscan operation using the passed data and MPI communicator. The input data will be overwritten
     * (inplace) during the reduction using operator op. Exscan combines all the communicated data-elements of lower-
     * ranked MPI processes then and excluding self (e.g. self rank=3, 0 op 1 op 2).
     *
     * @param data - The input and output parameter for the reduced data
     * @param op   - The MPI reduction operator handle (e.g. MPI_SUM)
     * @param comm - The MPI communicator to perform the reduction operation on
     * @returns The MPI error code
     */
    int exscan_inplace(af::array& data, MPI_Op op, MPI_Comm comm);

    /**
     * scan_inplace
     *
     * Refer to exscan_inplace, except own data-elements are included.
     *
     * @param data - The input and output parameter for the reduced data
     * @param op   - The MPI reduction operator handle (e.g. MPI_SUM)
     * @param comm - The MPI communicator to perform the reduction operation on
     * @returns The MPI error code
     */
    int scan_inplace(af::array& data, MPI_Op op, MPI_Comm comm);

    /**
     * inplace_reduction_collective
     *
     * Implementation meat for the inplace collective wrapper function such as allreduce_inplace, ... Receives a
     * function pointer to the actual MPI reduction collective operation and handles all the memory/buffer management
     * internally.
     *
     * @param data     - The input and output parameter for the reduced data
     * @param function - An MPI reduction collective function pointer (e.g. MPI_Allreduce, MPI_Exscan, ...)
     * @param op       - The MPI reduction operator handle (e.g. MPI_SUM)
     * @param comm     - The MPI communicator to perform the reduction operation on
     * @returns The MPI error code
     */
    int inplace_reduction_collective(af::array& data, ReductionCollective function, MPI_Op op, MPI_Comm comm);

} // namespace mpi
} // namespace juml

#endif //JUML_MPI_H
