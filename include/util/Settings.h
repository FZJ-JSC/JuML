/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: Settings.h
*
* Description: Header of class Settings
*
* Maintainer:
*
* Email:
*/

#ifndef SETTINGS_H
#define SETTINGS_H

#include <arrayfire.h>

namespace juml {
    //! Settings
    //! TODO: Describe me
    class Settings {
    protected:
        //! Settings constructor
        af::Backend backend_;    
        Settings();
        
    public:
        static Settings getInstance();
        
        void set_backend(af::Backend backend);
        af::Backend get_backend() const;
    }; // Settings
}  // juml
#endif // SETTINGS_H

