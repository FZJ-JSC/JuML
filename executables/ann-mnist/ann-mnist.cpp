#include <classification/ANN.h>
#include <mpi.h>
#include<arrayfire.h>
#include <iostream>

int main(int argc, char *argv[]) {
	using std::cout;
	using std::endl;
	MPI_Init(&argc, &argv);
	if (argc != 2 && argc != 3) return 1;
	af::setBackend(AF_BACKEND_CPU);
	af::info();
	if (argc == 3) {
		af::setSeed(atoi(argv[2]));
	}
	cout << "Seed: " << af::getSeed() << endl;

	juml::SequentialNeuralNet net(AF_BACKEND_CPU);
	//net.load("mnist-net.h5");
	net.add(juml::ann::make_SigmoidLayer(28*28, 100));
	net.add(juml::ann::make_SigmoidLayer(100, 50));
	net.add(juml::ann::make_SigmoidLayer(50, 10));
	//net.save("mnist-net.h5", false);

	//70000 x 28 x 28 will be read as 28 x 28 x 70000?
	juml::Dataset data(argv[1], "Data");
	juml::Dataset label(argv[1], "Label");

	int n_classes = 10;
	data.load_equal_chunks();
	label.load_equal_chunks();


	const int dataset_stepsize = 11;
	af::array data_array = data.data();
	data_array = af::moddims(data_array, 28*28, data_array.dims(2));
	data_array = data_array / 255.0;
	af::array full_data = data_array;

	//data_array = data_array(af::span, af::seq(0, data_array.dims(1) - 1, dataset_stepsize));
	data_array = data_array(af::span, af::seq(0, 8000 - 1));

	af::array label_array = label.data();
	//label_array = label_array(af::span, af::seq(0, label_array.dims(1) - 1, dataset_stepsize));
	label_array = label_array(af::span, af::seq(0, 8000 - 1));

	const int N = data_array.dims(1);
	int globalN;
	MPI_Allreduce(&N, &globalN, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	af::array target = af::constant(0, 10, N, s32);
	target(af::range(N).T() *  10 + label_array)= 1;

	int mpi_size;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	int batchsize = 100 / mpi_size;
	int nbatches = N/batchsize;
	cout << "N: " << N << " n_batches: " << nbatches << endl
		<< "batchsize: " << batchsize <<endl;
	for (int epoch = 0; epoch < 250; epoch++) {
		float error = 0;
		float lasterror;
		for (int batch = 0; batch < N; batch += batchsize) {
			int last = std::min((int)data_array.dims(1) - 1, batch + batchsize - 1);
			af::array batchsamples = data_array(af::span, af::seq(batch, last));
			af::array batchtarget = target(af::span, af::seq(batch, last));
			lasterror = net.fitBatch(batchsamples, batchtarget, 1);
			error += lasterror;
		}

		cout << " Epoch " << epoch << " Error: " << error / nbatches 
			<< " Last: " << lasterror << endl;
		cout << "Train-Class-Accuracy: " << (net.classify_accuracy_array(data_array, label_array) / (float) globalN) << endl;
		if (error / nbatches < 0.05) {
			break;
		}
	}
	
	juml::Dataset full_data_set(full_data);
	cout << "Full Class-Accuracy: " << net.classify_accuracy(full_data_set, label) << endl;
	/*juml::Dataset trainset(data_array);
	juml::Dataset labelset(target);
	net.fit(trainset, labelset);*/

	MPI_Finalize();
}
