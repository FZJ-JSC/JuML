#include<tclap/CmdLine.h>
#include <classification/ANN.h>
#include <mpi.h>
#include<arrayfire.h>
#include <iostream>
#include <vector>
#include <algorithm>

int main(int argc, char *argv[]) {
	using std::cout;
	using std::endl;
	MPI_Init(&argc, &argv);

	double time_start = MPI_Wtime();


	int mpi_size, mpi_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

	int n_classes = 52;
	int n_features = 30;
	int n_hidden_nodes = 16000;
	float LEARNINGRATE = 0.01;
	int batchsize = 400 / mpi_size;
	int max_epochs = 1000;
	std::string trainingFilePath;
	std::string xDatasetName = "Data";
	std::string yDatasetName = "Label";
	float max_error = 0.25;

	std::vector<int> hidden_layers;

	std::string networkFilePath;

	try {
		TCLAP::CmdLine cmd("Train a Artificial Neural Network for Classification", ' ', "0.1-alpha");

		TCLAP::ValueArg<int> cmd_classes("c", "classes", "The number of classes the classifier should be trained for.", true, 0, "Number of Classes", cmd);
		TCLAP::ValueArg<int> cmd_features("f", "features", "The number of features the data will contain", true, 0, "Number of Features", cmd);
		TCLAP::ValueArg<float> cmd_learningrate("l", "learningrate", "Learningrate for training of the ANN", false, 0.1, "Learningrate", cmd);
		TCLAP::MultiArg<int> cmd_hidden("", "hidden", "Add a hidden Layer with N nodes", true, "N", cmd);
		TCLAP::ValueArg<int> cmd_gbatchsize("b", "batchsize", "Set the global batchsize", true, 0, "Batchsize", cmd);
		//TCLAP::ValueArg<int> cmd_lbatchsize("B", "local-batchsize", "set teh local batchsize", true, 0, "Local batchsize", cmd);

		TCLAP::ValueArg<int> cmd_epochs("", "epochs", "Set the maximum number of training epochs", false, 1000, "Epochs", cmd);
		TCLAP::ValueArg<float> cmd_error("", "error", "Set the training error, after which to stop training", false, 0.25, "Error", cmd);
		TCLAP::ValueArg<std::string> cmd_datafile("d", "data", "Path to HDF5 file, that contains the training data", true, "", "PATH", cmd);
		TCLAP::ValueArg<std::string> cmd_datafile_dataset("", "data-set", "Name of the HDF5 dataset, that contains the training data", true, "Data", "Dataset Name", cmd);
		TCLAP::ValueArg<std::string> cmd_datafile_labelset("", "label-set", "Name of the HDF5 dataset, that contains the training labels", true, "Label", "Dataset Name", cmd);
		TCLAP::ValueArg<std::string> cmd_net("n", "net", "Path to store and read the ANN. If file is already present it will be read and training continued.", true, "", "PATH", cmd);

		n_classes = cmd_classes.getValue();
		n_features = cmd_features.getValue();
		LEARNINGRATE = cmd_learningrate.getValue();
		batchsize = cmd_gbatchsize.getValue() / mpi_size;
		max_epochs = cmd_epochs.getValue();
		max_error = cmd_error.getValue();

		hidden_layers = cmd_hidden.getValue();

		trainingFilePath = cmd_datafile.getValue();
		xDatasetName = cmd_datafile_dataset.getValue();
		yDatasetName = cmd_datafile_labelset.getValue();

		networkFilePath = cmd_net.getValue();

		cmd.parse(argc, argv);
	} catch (TCLAP::ArgException e) {
		std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
		MPI_Finalize();
		return 1;
	}



	if (argc != 2 && argc != 3) return 1;
	const af::Backend backend = AF_BACKEND_CUDA;
	cout << "CUDA_VISIBLE_DEVICES: " << secure_getenv("CUDA_VISIBLE_DEVICES") << endl;
	af::setBackend(backend);
	af::setDevice(mpi_rank % 4); // TODO: need to fix this
	cout << "Backend set" << endl;
	af::info();
	cout << "Seed: " << af::getSeed() << endl;

	double time_init_af = MPI_Wtime();

	juml::SequentialNeuralNet net(backend);

	{
		struct stat buffer;
		if (stat(networkFilePath.c_str(), &buffer) == 0) {
			// File exists
			net.load(networkFilePath);
			cout << "ANN loaded from file " << networkFilePath << endl;
			// TODO Check that ANN loaded from file matches specification from command line!
		} else {
			cout << "Creating new ANN" << endl;
			auto it = hidden_layers.begin();
			int previous_layer = *it;
			net.add(juml::ann::make_SigmoidLayer(n_features, *it));
			it++;
			for (; it != hidden_layers.end(); it++) {
				net.add(juml::ann::make_SigmoidLayer(previous_layer, *it));
				previous_layer = *it;
			}
			net.add(juml::ann::make_SigmoidLayer(previous_layer, n_classes));
		}
	}
	cout << "Learningrate: " << LEARNINGRATE << endl;

	// Print network layer counts:
	{
		auto it = net.layers_begin();
		cout << "Layers: " << (*it)->input_count;
		for (; it != net.layers_end(); it++) {
			cout << "-" << (*it)->node_count;
		}
		cout << endl;
	}

	double time_init_net = MPI_Wtime();

	juml::Dataset data(trainingFilePath, xDatasetName);
	juml::Dataset label(trainingFilePath, yDatasetName);

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
	if (mpi_rank == 0) {
		printf("%5s %10s %10s %10s\n", "Epoch", "Error", "Last Error", "Accuracy");
	}
	for (int epoch = 0; epoch < max_epochs; epoch++) {
		float error = 0;
		float lasterror;
		for (int batch = 0; batch < N; batch += batchsize) {
			int last = std::min(static_cast<int>(data_array.dims(1)) - 1, batch + batchsize - 1);
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
		float classify_accuracy = net.classify_accuracy_array(data_array, label_array) / static_cast<float>(globalN);
		if (mpi_rank == 0) {
			printf("%5d %10.6f %10.6f %10.6f\n", epoch, error/nbatches, lasterror, classify_accuracy);
		}
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

	net.save(networkFilePath, true);

	double time_saved = MPI_Wtime();
	cout << "Time Measuerment: " << endl;
	#define Fs(time, name) printf("%20s %3.4f\n", name, time);
	#define F(start, end, name) Fs(end - start, name);

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
