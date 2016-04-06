/*
 * Copyright (c) 2015
 * Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
 *
 * This software may be modified and distributed under the terms of BSD-style license.
 *
 * File name: Backend.cpp
 *
 * Description: Implementation of the Backend class
 *
 * Maintainer: m.goetz
 *
 * Email: murxman@gmail.com
 */

#include <arrayfire.h>

#include "core/Backend.h"

namespace juml {
    void Backend::set(int backend) {
        af::setBackend(static_cast<af::Backend>(backend));
    }

    int Backend::of(const af::array& data) {
        return af::getBackendId(data);
    }

    Backend::Backend(int backend) 
      : backend_(backend) {
      af::setBackend(static_cast<af::Backend>(this->backend_));
    }
        
    int Backend::get() const {
        return this->backend_;
    }
} // juml

