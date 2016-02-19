#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <arrayfire.h> 
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>


af::array* toAF(PyArrayObject* np_array, bool copy=true)
{  
    int ndims = PyArray_NDIM(np_array);
    if(ndims > 4)        
    {
        PyErr_SetString(PyExc_ValueError,"Not more than 4 dimensions suported");
        return NULL;
    }
    npy_intp dims[ndims];
    std::copy(PyArray_DIMS(np_array), PyArray_DIMS(np_array)+ndims,dims);
    bool c_style = PyArray_IS_C_CONTIGUOUS(np_array);
    if(c_style)
        std::swap(dims[0],dims[1]);
    af::dim4 dimensions(ndims,(dim_t*)dims);    
    af::dtype type;
    switch(PyArray_TYPE(np_array))
    {   
        case NPY_BOOL: type = b8; break;
        case NPY_INT16: type = s16; break;
        case NPY_INT32: type = s32; break;
        case NPY_INT64: type = s64; break;
        case NPY_UINT8: type = u8; break;
        case NPY_UINT16: type = u16; break;
        case NPY_UINT32: type = u32; break;
        case NPY_UINT64: type = u64; break;
        case NPY_FLOAT32: type = f32; break;
        case NPY_FLOAT64: type = f64; break;
        case NPY_COMPLEX64: type = c32; break;
        case NPY_COMPLEX128: type = c64; break;
        default : 
        {
            PyErr_SetString(PyExc_ValueError,"The provided type is not supported");
            return NULL;
        }
    }   
    
    char* data = (char*)PyArray_DATA(np_array);
    af::array* ret;
    if (copy || af::getBackendId(af::constant(0,1)) != AF_BACKEND_CPU || c_style)
    {
        ret = new af::array(dimensions, type);
        ret->write(data, PyArray_NBYTES(np_array));
        if(c_style)
        {  
            af::array transposed = ret->T();
            delete ret;
            ret = new af::array(transposed);
        }
    }
    else
    {
        af_array* arr;
        af_device_array(arr, data, ndims, (long long*)dims, type);
        ret = new af::array(arr);
        ret->lock();
    }
    return ret;
}

PyObject* toNumpy(af::array* af_array, bool copy = true)
{  
    int ndims = af_array->numdims();
    npy_intp dims[ndims];
    for (int i = 0; i < ndims; i++) 
        dims[i] = af_array->dims(i);
    
    int type;
    switch(af_array->type())
    {   
        case b8 : type = NPY_BOOL; break;
        case s16 : type = NPY_INT16; break;
        case s32 : type = NPY_INT32; break;
        case s64 : type = NPY_INT64; break;
        case u8 : type = NPY_UINT8; break;
        case u16 : type = NPY_UINT16; break;
        case u32 : type = NPY_UINT32; break;
        case u64 : type = NPY_UINT64; break;
        case f32 : type = NPY_FLOAT32; break;
        case f64 : type = NPY_FLOAT64; break;
        case c32 : type = NPY_COMPLEX64; break;
        case c64 : type = NPY_COMPLEX128; break;
        default : 
        {
            PyErr_SetString(PyExc_ValueError,"The provided type is not supported");
            return NULL;
        }
    }
    void* data;
    bool owner = true;
    if(!copy && af::getBackendId(*af_array) == AF_BACKEND_CPU)
    { 
        data = af_array->device<char>();
        owner = false;
    }
    else
    {
        data = new char[af_array->bytes()];
        af_array->host(data);
    }
    PyObject* array = PyArray_SimpleNewFromData(ndims, dims, type, data);
    if(owner)
        PyArray_ENABLEFLAGS((PyArrayObject*) array, NPY_ARRAY_OWNDATA);
    return array;
}
