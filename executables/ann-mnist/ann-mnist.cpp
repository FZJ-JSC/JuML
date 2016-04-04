#include <classification/ANN.h>
#include <mpi.h>
#include<arrayfire.h>
#include <iostream>

int main(int argc, char *argv[]) {
	using std::cout;
	using std::endl;
	MPI_Init(&argc, &argv);

	double time_start = MPI_Wtime();

	int mpi_size, mpi_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	if (argc != 2 && argc != 3) return 1;
	const af::Backend backend = AF_BACKEND_CUDA;
	cout << "CUDA_VISIBLE_DEVICES: " << secure_getenv("CUDA_VISIBLE_DEVICES") << endl;
	{
		char cuda_env[30];
		snprintf(cuda_env, 30, "CUDA_VISIBLE_DEVICES=%d", mpi_rank % 4);
		putenv(cuda_env);
	}
	cout << "CUDA_VISIBLE_DEVICES: " << secure_getenv("CUDA_VISIBLE_DEVICES") << endl;
	af::setBackend(backend);
	cout << "Backend set" << endl;
	af::info();
	if (argc == 3) {
		af::setSeed(atoi(argv[2]));
	}
	cout << "Seed: " << af::getSeed() << endl;

	double time_init_af = MPI_Wtime();

	juml::SequentialNeuralNet net(backend);
	//net.add(juml::ann::make_SigmoidLayer(28*28, 30));
	//net.add(juml::ann::make_SigmoidLayer(30, 10));
	//net.load("mnist-net.h5");
	net.add(juml::ann::make_SigmoidLayer(28*28, 100));
	net.add(juml::ann::make_SigmoidLayer(100, 50));
	net.add(juml::ann::make_SigmoidLayer(50, 10));
	//net.save("mnist-net.h5", false);a
	
	double time_init_net = MPI_Wtime();	

	//70000 x 28 x 28 will be read as 28 x 28 x 70000?
	juml::Dataset data(argv[1], "Data");
	juml::Dataset label(argv[1], "Label");

	int n_classes = 10;
	data.load_equal_chunks();
	label.load_equal_chunks();

	double time_loaded_data = MPI_Wtime();


	const int dataset_stepsize = 11;
	af::array data_array = data.data();
	data_array = af::moddims(data_array, 28*28, data_array.dims(2));
	data_array = data_array / 255.0;
	af::array full_data = data_array;

	//data_array = data_array(af::span, af::seq(0, data_array.dims(1) - 1, dataset_stepsize));
	data_array = data_array(af::span, af::seq(0, 8000/mpi_size - 1));

	af::array label_array = label.data();
	//label_array = label_array(af::span, af::seq(0, label_array.dims(1) - 1, dataset_stepsize));
	label_array = label_array(af::span, af::seq(0, 8000/mpi_size - 1));

	const int N = data_array.dims(1);
	int globalN;
	MPI_Allreduce(&N, &globalN, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	af::array target = af::constant(0, 10, N, s32);
	target(af::range(N).T() *  10 + label_array)= 1;

	double time_splitted_data = MPI_Wtime();
	double time_train_batch_test = 0;
	double time_train_sync = 0;

	int batchsize = 10;
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
			lasterror = net.fitBatch(batchsamples, batchtarget, 2);
			error += lasterror;
		}
		double time_buf = MPI_Wtime();
		net.sync();

		MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		error /= mpi_size;
		time_train_sync += MPI_Wtime() - time_buf;
		time_buf = MPI_Wtime();

		cout << " Epoch " << epoch << " Error: " << error / nbatches 
			<< " Last: " << lasterror << endl;
		cout << "Train-Class-Accuracy: " << (net.classify_accuracy_array(data_array, label_array) / (float) globalN) << endl;
		time_train_batch_test += MPI_Wtime() - time_buf;
		if (error / nbatches < 0.25) {
			break;
		}
	}

	double time_trained = MPI_Wtime();
	
	juml::Dataset full_data_set(full_data);
	cout << "Full Class-Accuracy: " << net.classify_accuracy(full_data_set, label) << endl;

	double time_tested = MPI_Wtime();
	/*juml::Dataset trainset(data_array);
	juml::Dataset labelset(target);
	net.fit(trainset, labelset);*/

	cout << "Time Measuerment: " << endl;
	#define Fs(time, name) printf("%20s %3.4f\n", name, time);
	#define F(start,end,name) Fs(end - start, name);

	F(time_start, time_init_af, "AF-init");	
	F(time_init_af, time_init_net, "Net-init");
	F(time_init_net, time_loaded_data, "Data load");
	F(time_loaded_data, time_splitted_data, "Split data");
	F(time_splitted_data, time_trained, "Training");
	Fs(time_train_batch_test, "Training-Test");
	Fs(time_train_sync, "Training-Sync");
	F(time_trained, time_tested, "Testing");
	F(time_start, time_tested, "Total");

	MPI_Finalize();
}
