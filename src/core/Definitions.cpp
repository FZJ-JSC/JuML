/*
 * Copyright (c) 2015
 * Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
 *
 * This software may be modified and distributed under the terms of BSD-style license.
 *
 * File name: Definitions.cpp
 *
 * Description: TODO
 *
 * Maintainer: m.glock
 *
 * Email: murxman@gmail.com
 */ 

#include "core/Definitions.h"

namespace juml {
    Backend::Backend(int backend) 
      : backend_(backend) { 
      af::setBackend(static_cast<af::Backend>(this->backend_));
    }
        
    int Backend::get() const {
        return this->backend_;
    }
} // juml

