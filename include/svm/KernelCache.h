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

#ifndef JUML_SVM_KERNELCACHE_H_
#define JUML_SVM_KERNELCACHE_H_
#include "Kernel.h"
#include <iostream>

namespace juml {

    namespace svm {

        //! KernelCache<Kernel>
        //! TODO: Describe me
        template <class Kernel> class KernelCache {
            const size_t max_bytes;
            Kernel& kernel;
            const unsigned int l;
            typedef decltype(kernel.evaluate_kernel(0,0)) kernel_t;

            // Cache Implementation inspired by libsvm
            // https://github.com/cjlin1/libsvm/blob/50415ead74a20b246b671fc6efdd11256e843207/svm.cpp#L61-L185
            // See LIBSVM-COPYRIGHT.txt
            // With modifications to fit our needs:
            // * Fixed Size buffer Cache without reallocations
            // * Cache a fixed number of columns instead of dynamic byte caching
            //   * This will make shrinking a lot less efficient, but i don't
            //     intend to implement shrinking right now.
            //     (With shrinking the kernel columns are swapped to the beginning of the set,
            //     which allows to make caching more efficient by only caching the first entries)

            unsigned int cachedColumns = 0;
            const unsigned int maxCachedColumns;
            kernel_t *data_space;

            struct head_t {
                head_t *prev, *next;
                kernel_t *data;
            };

            head_t *head;
            head_t lru_head;

            void lru_delete(head_t *h) {
                // delete from current location
                h->prev->next = h->next;
                h->next->prev = h->prev;
            }

            void lru_insert(head_t *h) {
                //insert to last position
                h->next = &lru_head;
                h->prev = lru_head.prev;
                h->prev->next = h;
                h->next->prev = h;
            }

            //! Request a column from cache.

            //! @param[in] index the index of the requested column
            //! @param[out] data pointer to where the column data is supposed to be stored or already partially stored
            //! @return `true` if there is already column data stored at the location specified by `data`
            bool get_data(const int index, kernel_t*& data) {
                head_t *h = &head[index];
                if (h->data) {
                    // There is already a data-pointer, so we already have that index cached.
                    // We need to move it to the front of the LRU list, therefor we delete it and add it again.
                    lru_delete(h);
                    lru_insert(h);

                    data = h->data;
                    return true;
                } else {
                    // The requested index is not cached
                    if (cachedColumns < maxCachedColumns) {
                        //The Cache is not full yet, so we don't need to replace anything
                        h->data = data_space + cachedColumns * l;
                        cachedColumns += 1;
                    } else {
                        //We need to replace
                        head_t *old = lru_head.next;
                        lru_delete(old);
                        //Assign the old buffer to the new entry
                        h->data = old->data;
                        old->data = nullptr;
                    }
                    lru_insert(h);

                    data = h->data;
                    return false;
                }
            }

#ifdef JUML_SVM_KERNELCACHE_STATISTIC
            unsigned long long colhit = 0;
            unsigned long long colmiss = 0;
            unsigned long long entryhit = 0;
            unsigned long long entrymiss = 0;
    #define COLHIT() colhit++
    #define COLMISS() colmiss++
    #define ENTRYHIT() entryhit++
    #define ENTRYMISS() entrymiss++
#else
    #define COLHIT() ;
    #define COLMISS() ;
    #define ENTRYHIT() ;
    #define ENTRYMISS() ;
#endif
            public:
                KernelCache(Kernel& kernel_, size_t bytes, unsigned int l_)
                    : kernel(kernel_),
                    max_bytes(bytes),
                    l(l_),
                    // how many Columns can we fit into `bytes` Bytes. Need minimum 2 Columns
                    maxCachedColumns(std::max<unsigned int>(bytes / (sizeof(kernel_t) * l), 2))
                {
                    head = new head_t[l];
                    for (int i = 0; i < l; i++) {
                        head[i].data = nullptr;
                    }
                    data_space = new kernel_t[maxCachedColumns * l];
                    lru_head.next = lru_head.prev = &lru_head;
                    std::cout<<"Cache has space for " << maxCachedColumns << " Kernel Columns" << std::endl;
                }

                ~KernelCache() {
#ifdef JUML_SVM_KERNELCACHE_STATISTIC
                    std::cout << "KernelCache deconstructed with: " << std::endl
                              << colhit << " Column Cache hits" << std::endl
                              << colmiss << " Column Cache misses" << std::endl
                              << (double)colhit/(colhit + colmiss)*100 << "% Column cache hit rate" << std::endl
                              << entryhit << " Entry Cache hits" << std::endl
                              << entrymiss << " Entry Cache misses" << std::endl
                              << (double)entryhit/(entryhit + entrymiss) * 100 << " % Entry hit rate" << std::endl;
#endif
                    delete[] head;
                    delete[] data_space;
                }

                const kernel_t* get_col(int col) {
                    kernel_t *data;
                    if (get_data(col, data)) {
                        //Column already partially cached
                        COLHIT();
                        for (int i = 0; i < l; i++) {
                            if (isnan(data[i])) {
                                ENTRYMISS();
                                data[i] = kernel.evaluate_kernel(i, col);
                            } else {
                                ENTRYHIT();
                            }
                        }
                    } else {
                        //Column is fresh, so we know we need to calculate everything
                        COLMISS();
                        for (int i = 0; i < l; i++) {
                            ENTRYMISS();
                            data[i] = kernel.evaluate_kernel(i, col);
                        }
                    }
                    return data;
                }

                const kernel_t* get_col(int col, std::vector<unsigned int> idxs) {
                    kernel_t *data;
                    if (get_data(col, data)) {
                        //Column already partially cached
                        COLHIT();
                        for (int i: idxs) {
                            if (isnan(data[i])) {
                                ENTRYMISS();
                                data[i] = kernel.evaluate_kernel(i, col);
                            } else {
                                ENTRYHIT();
                            }
                        }
                    } else {
                        //Column is fresh, so we know we need to calculate everythinga
                        COLMISS();
                        int lastindex = -1;
                        for (int i: idxs) {
                            //Need to fill the holes with NAN, so we know that we did not calculate these
                            std::fill(data + lastindex + 1, data + i, NAN);
                            ENTRYMISS();
                            data[i] = kernel.evaluate_kernel(i, col);
                            lastindex = i;
                        }
                        std::fill(data + lastindex + 1, data + l, NAN);
                    }
                    return data;
                }
        }; // KernelCache<Kernel>

#undef COLHIT
#undef COLMISS
#undef ENTRYHIT
#undef ENTRYMISS
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



#endif // JUML_SVM_KERNELCACHE_H_
