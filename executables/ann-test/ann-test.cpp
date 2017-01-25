#include <classification/ANN.h>
#include <mpi.h>
#include<arrayfire.h>
#include <iostream>

int main(int argc, char *argv[]) {
	using std::cout;
	using std::endl;
	MPI_Init(&argc, &argv);
	int mpi_size, mpi_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	if (argc != 3) return 1;
	const af::Backend backend = AF_BACKEND_CUDA;
 
 	af::setBackend(backend);
 	if (mpi_rank % 4 >= af::getDeviceCount()) {
 		cout << "Not enough devices to set device to " << (mpi_rank % 4) << " / " << af::getDeviceCount() << endl;
		return 1;
	}
	cout << "Backend set" << endl;
	af::setDevice(mpi_rank % 4);
	cout << "Device set to " << (mpi_rank % 4)  <<endl;
	af::info();

	juml::SequentialNeuralNet net(backend);
	cout << "Loading Net..." << endl;
	net.load(argv[1]);

	cout << "Loading Data " << endl;
	juml::Dataset data(argv[2], "Data");
	juml::Dataset label(argv[2], "Label");

	int n_classes = 10;
	data.load_equal_chunks();
	label.load_equal_chunks();
	cout << "Predicting..." << endl;
	
	af::array data_array = data.data();
	af::array label_array = label.data();
	label_array -= 1;

	const int N = data_array.dims(1);
	int globalN;
	MPI_Allreduce(&N, &globalN, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	cout << "N: " << N << endl;
	cout << "GlobalN: " << globalN << endl;
	int correct = net.classify_accuracy_array(data_array, label_array);
	cout << "Correctly Classified:" << correct << endl;
	cout << "Accuracy: " << (correct / (float)globalN) << endl;

	MPI_Finalize();
}

