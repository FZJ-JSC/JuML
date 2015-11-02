%module JuML

%{
    #define SWIG_FILE_WITH_INIT
    #include "data/Dataset.h"
    #include "preprocessing/ClassNormalizer.h"
    #include "classification/BaseClassifier.h"
    #include "classification/GaussianNaiveBayes.h"
    #include "svm/QMatrix.h"
    #include "svm/Kernel.h"
    #include "svm/BinarySVC.h"
%}

/* Wrap std::string */
%include std_string.i

/* Wrap mpi4py */
%include mpi4py/mpi4py.i
%mpi4py_typemap(Comm, MPI_Comm);

/*Wrap Armadillo*/
%include "armanpy.i"

/* Wrap Dataset*/
%include "data/Dataset.h"
%template(IntDataset) juml::Dataset<int>;
%template(FloatDataset) juml::Dataset<float>;

/* Wrap preprocessing */
%include "preprocessing/ClassNormalizer.h"

/* Wrap classification */
%include "classification/BaseClassifier.h"
%include "classification/GaussianNaiveBayes.h"

/* Wrap SVM */
%include "svm/QMatrix.h"
%include "svm/Kernel.h"
%include "svm/BinarySVC.h"

/* Remove unwanted *_swigregister globals */
%pythoncode %{
def __cleanup_namespace():
    for i in globals().copy():
        if i.endswith("_swigregister"):
            del globals()[i]
__cleanup_namespace()
del __cleanup_namespace
%}
