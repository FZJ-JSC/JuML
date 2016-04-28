/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: Dataset.h
*
* Description: Header of class Dataset
*
* Maintainer: m.goetz
*
* Email: murxman@gmail.com
*/

#ifndef DATASET_H
#define DATASET_H

#include <arrayfire.h>
#include <hdf5.h>
#include <mpi.h>
#include <string>
#include <sys/stat.h>

namespace juml {
    /**
     * Dataset
     *
     * This class implements a distributed data container that allows to load and store data from HDF5 files and in-
     * memory arrays. It is used in algorithms as a mean to pass data samples and labels and capture prediction results.
     */
    class Dataset {
    protected:
        /**
         * @var   data_
         * @brief An arrayfire containing the local portion of the entire data of this node
         */
        af::array data_;

        /**
         * @var   filename_
         * @brief The name of the HDF5 file the data is located in
         */
        const std::string filename_;
        /**
         * @var   dataset_
         * @brief The name of the HDF5 dataset the data is located in
         */
        const std::string dataset_;

        /**
         * @var   loading_time_
         * @brief The UNIX time stamp when the dataset has been loaded from disk, defaults to 0
         */
        time_t loading_time_ = 0;

        /**
         * @var   comm_
         * @brief The MPI communicator the data is distributed over
         */
        const MPI_Comm comm_;
        /**
         * @var   mpi_rank_
         * @brief The node's MPI rank in comm_
         */
        int mpi_rank_;
        /**
         * @var   mpi_size_
         * @brief The size of comm_
         */
        int mpi_size_;

        /**
         * @var   sample_dim_
         * @brief The dimension index which contains the samples, usually the highest dimension, e.g. for 2D data the
         *        sample dimension is 1 or for 3D data 2.
         */
        dim_t sample_dim_;
        /**
         * @var   global_n_samples_
         * @brief The number of samples distributed across all nodes
         */
        dim_t global_n_samples_;
        /**
         * @var   global_offset_
         * @brief The global sample offset of the local data portion, only meaningful for consecutive data chunks
         */
        dim_t global_offset_;

        /**
         * h5_to_af
         *
         * Finds the arrayfire type equivalent for an HDF5 type handle (e.g. H5T_STD_I32LE -> s32)
         *
         * @param h5_type - The HDF5 type handle
         * @returns The arrayfire type equivalent
         * @throws domain_error if there is no conversion equivalent
         */
        af::dtype h5_to_af(hid_t h5_type);

        /**
         * af_to_h5
         *
         * Converts a arrayfire type to an HDF5 type handle (e.g. s32 -> H5T_I32LE)
         *
         * @param af_type - The arrayfire type
         * @returns The HDF5 type handle equivalent
         * @throws domain_error if there is no conversion equivalent
         */
        hid_t af_to_h5(af::dtype af_type);

    public:
        /**
         * Dataset constructor
         *
         * Creates a new dataset from an HDF5 file.
         *
         * @param filename - The name of the HDF5 file to load
         * @param dataset  - The HDF5 dataset name
         * @param comm     - The MPI comm the data will be distributed across
         */
        Dataset(const std::string& filename, const std::string& dataset, const MPI_Comm comm=MPI_COMM_WORLD);
        /**
         * Dataset constructor
         *
         * Creates a new dataset from in-memory data. It is considered to be the local portion of the global data.
         *
         * @param filename - The name of the HDF5 file to load
         * @param dataset  - The HDF5 dataset name
         * @param comm     - The MPI comm the data will be distributed across
         */
        Dataset(const af::array& data, MPI_Comm comm=MPI_COMM_WORLD);

        /**
         * load_equal_chunks
         *
         * Actually loads data from a HDF5 file on disk. The file and dataset are set in the constructor. Data will only
         * be loaded once, unless it has changed on disk. The data will be load in equal portions/chunks on each of the
         * nodes in the set MPI communicator. Each of the portion are consecutive in the HDF5 file.
         *
         * @param force - Force the load data from disk, even if it has not been modified since the initial load
         * @throws runtime_error if the file or dataset does not exist or cannot be accessed
         * @throws domain_error  if the data in the HDF5 has more then four dimensions
         */
        void load_equal_chunks(bool force=false);

        /**
         * dump_equal_chunks
         *
         * Stores data in the dataset on the disk in an HDF5 file. The data is assumed to be consecutive, offset by the
         * multiples of the chunk sizes of each MPI node's rank in the communicator.
         *
         * @param filename - The name of the HDF5 file to store the data in, will be created if it does not exist.
         * @param dataset  - The name of the HDF5 dataset to store the data in
         */
        void dump_equal_chunks(const std::string& filename, const std::string& dataset);

        /**
         * mean
         *
         * Calculates the mean/average value of the dataset
         *
         * @param total - True if the mean should be calculated locally or false of globally, defaults to false
         */
        af::array mean(bool total=false) const;
        /**
         * stddev
         *
         * Calculates the standard deviation of the dataset
         *
         * @param total - True if the mean should be calculated locally or false of globally, defaults to false
         */
        af::array stddev(bool total=false) const;
        /**
         * normalize
         *
         * Normalizes the features of the dataset to a predefined interval (usually 0-1).
         *
         * @param min                  - The lower boundary of the normalization interval, defaults to 0
         * @param max                  - The upper boundary of the normalization interval, defaults to 1
         * @param independent_features - Determines whether each feature column is normalized separately all the whole
         *                               numerical range of all features is considered, defaults to true (independent)
         * @param selected_features    - A vector with feature indices to normalize, defaults to empty (all features)
         * @throws invalid_argument if selected features is not a vector but a matrix
         */
        void normalize(float min=0, float max=1, bool independent_features=true,
                       const af::array& selected_features = af::array());
        /**
         * normalize_stddev
         *
         * Normalized the features of the dataset to multiples of the feature's standard deviation.
         *
         * @param x_std                - The multiple of standard deviations that is normalized to one, defaults to one
         * @param independent_features - Determines whether each feature column is normalized separately all the whole
         *                               numerical range of all features is considered, defaults to true (independent)
         * @param selected_features    - A vector with feature indices to normalize, defaults to empty (all features)
         * @throws invalid_argument if selected features is not a vector but a matrix, or x_std is less than zero
         */
        void normalize_stddev(float x_std=1, bool independent_features=true,
                              const af::array &selected_features=af::array());

        /**
         * loading_time
         *
         * @returns The UNIX timestamp when the data was loaded from disk, zero if not loaded
         */
        time_t loading_time() const;
        /**
         * modified_time
         *
         * @returns The UNIX timestamp when the HDF5 backing file was last modified on disk
         */
        time_t modified_time() const;

        virtual af::array& data();
        virtual const af::array& data() const;
        virtual dim_t n_samples() const;
        virtual dim_t n_features() const;
        virtual dim_t global_n_samples() const;
        virtual dim_t global_offset() const;
        virtual dim_t sample_dim() const;
    }; // Dataset
}  // juml
#endif // DATASET_H
