#include <arrayfire.h> 
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

af::array* toAF(PyArrayObject* np_array)
{  
    int ndims = PyArray_NDIM(np_array);
    if(ndims > 4)        
    {
        PyErr_SetString(PyExc_ValueError,"Not more than 4 dimensions suported");
        return NULL;
    }
    npy_intp dims[ndims];
    std::copy(PyArray_DIMS(np_array), PyArray_DIMS(np_array)+ndims,dims);
    if(PyArray_IS_C_CONTIGUOUS(np_array))
        std::swap(dims[0],dims[1]);
    af::dim4 dimensions(ndims,(dim_t*)dims); 
    char* data = (char*)PyArray_DATA(np_array);
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
    af::array* ret = new af::array(dimensions, type);
    ret->write(data, PyArray_NBYTES(np_array));
    if(PyArray_IS_C_CONTIGUOUS(np_array))
    {   af::array transposed = ret->T();
        delete ret;
        ret = new af::array(transposed);
    }
    return ret;
}

PyObject* toNumpy(af::array* af_array)
{  
    int ndims = af_array->numdims();
    npy_intp dims[ndims];
    for (int i = 0; i < ndims; i++) 
        dims[i] = af_array->dims(i);
    
    void* data = new char[af_array->bytes()];
    af_array->host(data);
    
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
    PyObject* array = PyArray_SimpleNewFromData(ndims, dims, type, data);
    PyArray_ENABLEFLAGS((PyArrayObject*) array, NPY_OWNDATA);
    return array;
}
