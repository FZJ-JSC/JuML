%module JuML

%{
    #define SWIG_FILE_WITH_INIT
    #include "core/Algorithm.h"
    #include "data/Dataset.h"
    #include "preprocessing/ClassNormalizer.h"
    #include "classification/BaseClassifier.h"
    #include "classification/GaussianNaiveBayes.h"
%}

/* Convert C++ exceptions to Python exception */
%exception {
   try {
      $action
   } catch (std::exception &e) {
      PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
      return NULL;
   }
}
/* Wrap std::string */
%include std_string.i

/* Wrap mpi4py */
%include mpi4py/mpi4py.i
%mpi4py_typemap(Comm, MPI_Comm);

/* Wrap arrayfire */
%include arrayfire.i

/*Wrap Core*/
%include "core/Algorithm.h"
%include "core/Backend.h"

/* Wrap Dataset*/
%include "data/Dataset.h"

/* Wrap preprocessing */
%include "preprocessing/ClassNormalizer.h"

/* Wrap classification */
%include "classification/BaseClassifier.h"
%include "classification/GaussianNaiveBayes.h"

/* Remove unwanted *_swigregister globals */
%pythoncode %{
def __cleanup_namespace():
    for i in globals().copy():
        if i.endswith("_swigregister"):
            del globals()[i]
__cleanup_namespace()
del __cleanup_namespace
%}
