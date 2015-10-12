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

namespace juml {
    namespace svm {
        template <class Kernel> class KernelCache {
            const size_t max_bytes;
            Kernel& kernel;
            const unsigned int n;
            typedef decltype(kernel.evaluate_kernel(0,0)) kernel_t;
            arma::Col<kernel_t> stub;
            public:
                KernelCache(Kernel& kernel_, size_t bytes, unsigned int n_) : kernel(kernel_), max_bytes(bytes), n(n_), stub(n) {}
                const kernel_t* get_col(int col, std::vector<unsigned int> idxs) {
                    for (auto z: idxs) {
                        stub(z) = kernel.evaluate_kernel(z, col);
                    }
                    return stub.memptr();
                }
        };
    }
}



#endif // KERNEL_CACHE_H
