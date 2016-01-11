/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: Settings.cpp
*
* Description: Implementation of class Settings
*
* Maintainer:
*
* Email:
*/

#include "util/Settings.h"

namespace juml {

    Settings::Settings() {
        this->set_backend(AF_BACKEND_DEFAULT);
    }

    Settings Settings::getInstance() {
        static Settings INSTANCE;
        return INSTANCE;
    }
    
    void Settings::set_backend(af::Backend backend) {
        this->backend_ = backend;
        af::setBackend(backend);
    }
    
    af::Backend Settings::get_backend() const {
        return this->backend_;        
    }
    
} // namespace juml

