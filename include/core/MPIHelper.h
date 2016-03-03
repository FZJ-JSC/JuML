#ifndef JUML_MPIHELPER_INCLUDED_
#define JUML_MPIHELPER_INCLUDED_

#include <mpi.h>
#include <arrayfire.h>
namespace juml {
	namespace mpi {
		inline bool canUseDevicePoiner(const af::array& a) {
			return af::getBackendId(a) == AF_BACKEND_CPU;
		}

		template<typename T>
		MPI_Datatype getMpiType() {
			return -1;
		}
		template<>
		MPI_Datatype getMpiType<float>() {
			return MPI_FLOAT;
		}

		template<typename T>
		void allreduceInplace(af::array& a, MPI_Op op, MPI_Comm comm) {
			T* ptr;
			if (canUseDevicePoiner(a)) {
				ptr = a.device<T>();
			} else {
				ptr = (T*)malloc(sizeof(T) * a.elements());
				a.host(ptr);
			}
			MPI_Allreduce(MPI_IN_PLACE, ptr, a.elements(), getMpiType<T>(), op, comm);
			if (canUseDevicePoiner(a)) {
				a.unlock();
			} else {
				free(ptr);
			}
		}
	}
}
#endif
