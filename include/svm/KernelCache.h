// vim: expandtab:shiftwidth=4:softtabstop=4
/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: KernelCache.h
*
* Description: Header of class KernelCache
*
* Maintainer:
*
* Email:
*/

#ifndef KERNEL_CACHE_H
#define KERNEL_CACHE_H
#include "Kernel.h"

namespace juml {

    namespace svm {

        //! KernelCache<Kernel>
        //! TODO: Describe me
        template <class Kernel> class KernelCache {
            const size_t max_bytes;
            Kernel& kernel;
            const unsigned int n;
            typedef decltype(kernel.evaluate_kernel(0,0)) kernel_t;
            arma::Col<kernel_t> stub;
            public:
                KernelCache(Kernel& kernel_, size_t bytes, unsigned int n_) : kernel(kernel_), max_bytes(bytes), n(n_), stub(n) {}

                const kernel_t* get_col(int col) {
                    for (int row = 0; row < n; row++) {
                        stub(row) = kernel.evaluate_kernel(row, col);
                    }
                    return stub.memptr();
                }

                const kernel_t* get_col(int col, std::vector<unsigned int> idxs) {
                    for (auto z: idxs) {
                        stub(z) = kernel.evaluate_kernel(z, col);
                    }
                    return stub.memptr();
                }
        }; // KernelCache<Kernel>


        //! KernelCache<Kernel<KernelType::PRECOMPUTED, kernel_t>>
        //! TODO: Describe me
        template<typename kernel_t> class KernelCache<Kernel<KernelType::PRECOMPUTED, kernel_t>> {
            const Kernel<KernelType::PRECOMPUTED, kernel_t> kernel;
            public:
                KernelCache(const Kernel<KernelType::PRECOMPUTED, kernel_t>& kernel_) : kernel(kernel_) {}

                inline const kernel_t* get_col(int col) {
                    return kernel.precomputed_kernel.colptr(col);
                }

                inline const kernel_t* get_col(int col, std::vector<unsigned int> idxs) {
                    return kernel.precomputed_kernel.colptr(col);
                }

        }; // KernelCache<Kernel<KernelType::PRECOMPUTED, kernel_t>>
    }
}



#endif // KERNEL_CACHE_H
