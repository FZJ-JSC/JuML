%{
#include <arrayfire_converter.h>
%} 
%init %{
import_array();
%}

%typemap(typecheck, precedence=SWIG_TYPECHECK_FLOAT_ARRAY)
( const af::array & ),
(       af::array & )
{
   $1 = PyArray_Check($input) ? 1 : 0;
}

%typemap(in)
( const af::array & ),
(       af::array & ) 
{
    PyArrayObject *temp=NULL;
    if (PyArray_Check($input))
        temp = (PyArrayObject*)$input; 
    else    
    {
        PyErr_SetString(PyExc_ValueError,"Input object is not an array");
        return NULL;
    }
    $1=toAF(temp);
}

%typemap(freearg)
( const af::array & ),
(       af::array & ) 
{
   free($1);
}

%typemap(out)
( const af::array & ),
(       af::array & ) 
{
    $result = toNumpy($1);
}
