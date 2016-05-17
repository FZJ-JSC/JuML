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

	const int n_classes = 52;
	const int n_features = 30;
	const int n_hidden_nodes = 16000;
	const float LEARNINGRATE=0.01;
	const int batchsize = 400 / mpi_size;
	const int max_epochs = 1000;
	const char *xDatasetName = "Data";
	const char *yDatasetName = "Label";
	const float max_error = 0.25;

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
	cout << "Seed: " << af::getSeed() << endl;

	double time_init_af = MPI_Wtime();

	juml::SequentialNeuralNet net(backend);

	{
		struct stat buffer;
		if (argc == 3 && stat(argv[2], &buffer) == 0) {
			//File exists
			net.load(argv[2]);
			cout << "ANN loaded from file " << argv[2] << endl;
		} else {
			cout << "Creating new ANN" << endl;
			net.add(juml::ann::make_SigmoidLayer(n_features, n_hidden_nodes));
			net.add(juml::ann::make_SigmoidLayer(n_hidden_nodes, n_classes));
		}

	}
	cout << "Learningrate: " << LEARNINGRATE << endl;
	
	//Print network layer counts:
	{
		auto it = net.layers_begin();
		cout << "Layers: " << (*it)->input_count;
		for (;it != net.layers_end(); it++) {
			cout << "-" << (*it)->node_count;
		}
		cout << endl;
	}	
	
	double time_init_net = MPI_Wtime();	

	juml::Dataset data(argv[1], xDatasetName);
	juml::Dataset label(argv[1], yDatasetName);

	data.load_equal_chunks();
	label.load_equal_chunks();

	double time_loaded_data = MPI_Wtime();


	const int dataset_stepsize = 11;
	af::array data_array = data.data().as(f32);
	af::array full_data = data_array;

	//data_array = data_array(af::span, af::seq(0, data_array.dims(1) - 1, dataset_stepsize));
	//data_array = data_array(af::span, af::seq(0, 8000/mpi_size - 1));

	af::array label_array = label.data().as(s32) - 1;
	//label_array = label_array(af::span, af::seq(0, label_array.dims(1) - 1, dataset_stepsize));
	//label_array = label_array(af::span, af::seq(0, 8000/mpi_size - 1));

	const int N = data_array.dims(1);
	int globalN;
	MPI_Allreduce(&N, &globalN, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	af::array target = af::constant(0, n_classes, N, s32);
	target(af::range(N).T() *  n_classes + label_array)= 1;

	double time_splitted_data = MPI_Wtime();
	double time_train_batch_test = 0;
	double time_train_sync = 0;

	int nbatches = N/batchsize;
	cout << "N: " << N << " n_batches: " << nbatches << endl
		<< "batchsize: " << batchsize <<endl;
	printf("%5s %10s %10s %10s\n", "Epoch", "Error", "Last Error", "Accuracy");
	for (int epoch = 0; epoch < max_epochs; epoch++) {
		float error = 0;
		float lasterror;
		for (int batch = 0; batch < N; batch += batchsize) {
			int last = std::min((int)data_array.dims(1) - 1, batch + batchsize - 1);
			af::array batchsamples = data_array(af::span, af::seq(batch, last));
			af::array batchtarget = target(af::span, af::seq(batch, last));
			lasterror = net.fitBatch(batchsamples, batchtarget, LEARNINGRATE);
			error += lasterror;
		}
		double time_buf = MPI_Wtime();

		MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		error /= mpi_size;
		time_train_sync += MPI_Wtime() - time_buf;
		time_buf = MPI_Wtime();

		printf("%5d %10.6f %10.6f %10.6f\n", epoch, error/nbatches, lasterror, net.classify_accuracy_array(data_array, label_array) / (float) globalN);
		time_train_batch_test += MPI_Wtime() - time_buf;
		if (error / nbatches < max_error) {
			break;
		}
	}

	double time_trained = MPI_Wtime();
	
	juml::Dataset test_data_set(full_data);
	juml::Dataset test_label_set(label_array);
	cout << "Full Class-Accuracy: " << net.classify_accuracy(test_data_set, test_label_set) << endl;

	double time_tested = MPI_Wtime();
	/*juml::Dataset trainset(data_array);
	juml::Dataset labelset(target);
	net.fit(trainset, labelset);*/
	if (argc == 3) {
		net.save(argv[2], true);
		cout << "Saved ANN to " << argv[2] << endl;
	}
	double time_saved = MPI_Wtime();
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
	F(time_tested, time_saved, "Saving Net");
	F(time_start, time_saved, "Total");

	MPI_Finalize();
}