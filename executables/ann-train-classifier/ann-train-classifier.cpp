#include <classification/ANN.h>
#include <mpi.h>
#include<arrayfire.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "optionparser.h"
struct Arg: public option::Arg
{
   static void printError(const char* msg1, const option::Option& opt, const char* msg2)
   {
     fprintf(stderr, "ERROR: %s", msg1);
     fwrite(opt.name, opt.namelen, 1, stderr);
     fprintf(stderr, "%s", msg2);
   }

   static option::ArgStatus Unknown(const option::Option& option, bool msg)
   {
     if (msg) printError("Unknown option '", option, "'\n");
     return option::ARG_ILLEGAL;
   }

   static option::ArgStatus Required(const option::Option& option, bool msg)
   {
     if (option.arg != 0)
       return option::ARG_OK;

     if (msg) printError("Option '", option, "' requires an argument\n");
     return option::ARG_ILLEGAL;
   }

   static option::ArgStatus NonEmpty(const option::Option& option, bool msg)
   {
     if (option.arg != 0 && option.arg[0] != 0)
       return option::ARG_OK;

     if (msg) printError("Option '", option, "' requires a non-empty argument\n");
     return option::ARG_ILLEGAL;
   }

   static option::ArgStatus Numeric(const option::Option& option, bool msg)
   {
     char* endptr = 0;
     if (option.arg != 0 && strtol(option.arg, &endptr, 10)){};
     if (endptr != option.arg && *endptr == 0)
       return option::ARG_OK;

     if (msg) printError("Option '", option, "' requires an integer argument\n");
     return option::ARG_ILLEGAL;
   }
   static option::ArgStatus Float(const option::Option& option, bool msg)
   {
	   char *endptr = 0;
	   if (option.arg != 0 && strtof(option.arg, &endptr)){};
	   if (endptr != option.arg && *endptr ==0) {
		   return option::ARG_OK;
	   }
	   if (msg) printError("Option '", option, "' requires a numeric argument\n");
	   return option::ARG_ILLEGAL;
   }
   static option::ArgStatus ExistingFile(const option::Option& option, bool msg) {
	struct stat buffer;
	if (option.arg != 0 && stat(option.arg, &buffer) == 0) {
		return option::ARG_OK;
	}
	if (msg) printError("Option '", option, "' requires a file argument\n");
	return option::ARG_ILLEGAL;
   }

   template<option::ArgStatus F(const option::Option&, bool)>
   static option::ArgStatus OptionalF(const option::Option& option, bool msg)
   {
     if (option.arg == 0) return option::ARG_IGNORE;
     return F(option, msg);
   }
};

enum optionIndex{O_UNKNOWN, O_HELP, O_FEATURES, O_CLASSES, O_LEARNINGRATE, O_HIDDEN, O_BATCHSIZE, O_EPOCHS, O_MAXERROR,
	O_DATAFILE, O_DATAFILE_DATA_SET, O_DATAFILE_LABEL_SET, O_SEED, O_BACKEND, O_SYNCTYPE, O_NETFILE, O_SHUFFLE};

std::vector<int> requiredOptions = {O_FEATURES, O_CLASSES, O_LEARNINGRATE, O_BATCHSIZE, O_MAXERROR, O_DATAFILE, O_NETFILE, O_BACKEND};

//TODO: Document required arguments in USAGE
const option::Descriptor usage[] = {
	{O_UNKNOWN, 0, "", "", Arg::Unknown, "Train a Artificial Neural Network for Classification\n\n"
		"USAGE: juml-ann-train-classifier [options]\n\nGeneral Options:"},
	{O_HELP, 0, "h", "help", option::Arg::None, "--help, -h\tPrint usage and exit."},
	{O_SEED, 0, "", "seed", Arg::Numeric, "--seed <N>\tSet the seed used for random initialization"},
	{O_BACKEND, 1, "", "cpu", option::Arg::None, "--cpu \tUse the ArrayFire CPU Backend"},
	{O_BACKEND, 2, "", "opencl", option::Arg::None, "--opencl\tUse the ArrayFire OpenCL Backend"},
	{O_BACKEND, 3, "", "cuda", option::Arg::None, "--cuda \tUse the ArrayFire Cuda Backend"},

	{O_MAXERROR, 0, "", "error", Arg::Float, "--error <E>\tSet the training error, after which to stop training"},
	{O_EPOCHS, 0, "", "epochs", Arg::Numeric, "--epochs <N>\tSet the maximum number of training epochs"},
	{O_BATCHSIZE, 0, "b", "batchsize", Arg::Numeric, "--batchsize <B>, -b <B>\tSet the global batchsize."},
	{O_SHUFFLE, 0, "","shuffle-samples", option::Arg::None, "--shuffle-samples\tChange the order of the samples for each epoch"},
	{O_LEARNINGRATE, 0, "l", "learningrate", Arg::Float, "--learningrate <L>, -l <L>\tLearningrate for training of the ANN"},
	{O_SYNCTYPE, 1, "", "sync-after-batch", option::Arg::None, "--sync-after-batch\tSyncronize the ANN after each batch."},
	{O_SYNCTYPE, 0, "", "sync-after-epoch", option::Arg::None, "--sync-after-epoch\tSyncronize the ANN after each epoch."},
	{O_UNKNOWN, 0, "", "", NULL, 0},
	{O_UNKNOWN, 0, "", "", Arg::Unknown, "ANN-Options:"},
	{O_FEATURES, 0, "f", "features", Arg::Numeric, "--features <F>, -f <F>\tThe number of features the data will contain. The same as the number of neurons in the input layer."},
	{O_HIDDEN, 0, "H", "hidden", Arg::Numeric, "--hidden <N>, -H <N>\tAdd a hidden layer to the ANN."},
	{O_CLASSES, 0, "c", "classes", Arg::Numeric, "--classes <C>, -c <C>\tThe number of classes in the data. The same as the number of output neurons in the last layer."},

	{O_UNKNOWN, 0, "", "", NULL, 0},
	{O_UNKNOWN, 0, "", "", Arg::Unknown, "Input and Output Options:"},
	{O_DATAFILE, 0, "d", "data", Arg::ExistingFile, "--data PATH, -d PATH\tPath to HDF5 File, taht contains the training data"},
	{O_DATAFILE_DATA_SET, 0, "", "data-set", Arg::NonEmpty, "--data-set <S>\tSet the name of the Dataset inside the HDF5 data file that contains the features"},
	{O_DATAFILE_LABEL_SET, 0, "", "label-set", Arg::NonEmpty, "--label-set <S>\tSet the name of the Dataaset inside the HDF5 data file that contains the class labels"},
	{O_NETFILE, 0, "n", "net", Arg::Required, "--net <PATH>, -n <PATH>\tPath to a HDF5 file where the ANN will be loaded from and stored to. If the file does not exist a new ANN will be created"},
	{0, 0, 0, 0, 0, 0}
};

void printListOfOrOptions(int index) {
	int i = 0;
	bool foundMoreThanOne = false;
	bool isShortOption;
	const char *previous = NULL;
	while (true) {
		if (usage[i].index == 0 && usage[i].type == 0 && usage[i].shortopt == 0 && usage[i].longopt == 0 && usage[i].check_arg == 0 && usage[i].help == 0) {
			break;
		}
		if (usage[i].index == index) {
			if (previous == NULL) {
				isShortOption = usage[i].longopt[0] == 0;
				previous = isShortOption ? usage[i].shortopt : usage[i].longopt;
			} else {
				if (!foundMoreThanOne) {
					foundMoreThanOne = true;
					fprintf(stderr, "%s%s", isShortOption ? "-" : "--", previous);
				} else {
					fprintf(stderr, ", %s%s", isShortOption ? "-" : "--", previous);
				}
				isShortOption = usage[i].longopt[0] == 0;
				previous = isShortOption ? usage[i].shortopt : usage[i].longopt;
			}
		}
		i += 1;
	}
	if (previous != NULL) {
		if (foundMoreThanOne) fprintf(stderr, " or %s%s", isShortOption ? "-" : "--", previous);
		else fprintf(stderr, "%s%s", isShortOption ? "-" : "--", previous);
	}
}


int main(int argc, char *argv[]) {
	using std::cout;
	using std::endl;
	MPI_Init(&argc, &argv);

	double time_start = MPI_Wtime();


	int mpi_size, mpi_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

	int n_classes;
	int n_features;
	int n_hidden_nodes;
	float LEARNINGRATE;
	int batchsize;
	int max_epochs;
	std::string trainingFilePath;
	std::string xDatasetName;
	std::string yDatasetName;
	float max_error;
	int seed;
	af::Backend backend = AF_BACKEND_CUDA;
	bool sync_after_batch_update;
	bool shuffle_samples = false;

	std::vector<int> hidden_layers;

	std::string networkFilePath;

	{
		option::Stats stats(usage, argc - 1, argv + 1);
		std::vector<option::Option> options(stats.options_max);
		std::vector<option::Option> buffer(stats.buffer_max);
		option::Parser parse(usage, argc - 1, argv + 1, &options[0], &buffer[0]);

		if (parse.error()) {
			return 1;
		}
		if (argc != 1) {
			for (auto it = requiredOptions.begin(); it != requiredOptions.end(); it++) {
				if (!options[*it]) {
					fprintf(stderr, "ERROR: Missing required Argument: ");
					printListOfOrOptions(*it);
					fprintf(stderr, "\n");
					argc = 1;
				} else if (options[*it].count() > 1) {
					fprintf(stderr, "ERROR: Only specify once: ");
					printListOfOrOptions(*it);
					fprintf(stderr, "\n");
					argc = 1;
				}
			}
		}

		if (parse.nonOptionsCount() > 0) {
			fprintf(stderr, "ERROR: Trailing arguments\n");
			argc = 1;
		}

		if (options[O_HELP] || argc == 1) {
			option::printUsage(std::cout, usage);
			return 0;
		}

		n_classes = atoi(options[O_CLASSES].arg);
		n_features = atoi(options[O_FEATURES].arg);
		LEARNINGRATE = atof(options[O_LEARNINGRATE].arg);
		batchsize = atoi(options[O_BATCHSIZE].arg) / mpi_size;
		if (options[O_EPOCHS])
			max_epochs = atoi(options[O_EPOCHS].arg);
		else
			max_epochs = 1000;
		max_error = atof(options[O_MAXERROR].arg);


		for (option::Option* opt = options[O_HIDDEN]; opt; opt = opt->next()) {
			hidden_layers.push_back(atoi(opt->arg));
		}

		trainingFilePath = options[O_DATAFILE].arg;
		if (options[O_DATAFILE_DATA_SET])
			xDatasetName = options[O_DATAFILE_DATA_SET].arg;
		else
			xDatasetName = "Data";
		if (options[O_DATAFILE_LABEL_SET])
			yDatasetName = options[O_DATAFILE_LABEL_SET].arg;
		else
			yDatasetName = "Label";

		networkFilePath = options[O_NETFILE].arg;
		if (options[O_SEED]) 
			seed = atoi(options[O_SEED].arg);

		switch(options[O_BACKEND].last()->type()) {
			case 1: backend = AF_BACKEND_CPU; break;
			case 2: backend = AF_BACKEND_OPENCL; break;
			case 3: backend = AF_BACKEND_CUDA; break;
			default:
				fprintf(stderr, "Could not parse backend Argument to Backend\n");
		}
		if (options[O_SYNCTYPE]) {
			if (options[O_SYNCTYPE].count() > 1) {
				fprintf(stderr, "Can only specify one of ");
				printListOfOrOptions(O_SYNCTYPE);
				fprintf(stderr, "\n");
				return 1;
			}
			sync_after_batch_update = options[O_SYNCTYPE].last()->type();
		} else {
			sync_after_batch_update = true;
		}

		if (options[O_SHUFFLE]) {
			shuffle_samples = true;
		}

	}


	cout << "CUDA_VISIBLE_DEVICES: " << secure_getenv("CUDA_VISIBLE_DEVICES") << endl;
	af::setBackend(backend);
	af::setDevice(mpi_rank % 4); // TODO: need to fix this
	cout << "Backend set" << endl;
	af::info();
	af::setSeed(seed);
	cout << "Seed: " << af::getSeed() << endl;

	double time_init_af = MPI_Wtime();

	juml::SequentialNeuralNet net(backend);

	{
		cout << "Creating ANN" << endl;
		auto it = hidden_layers.begin();
		int previous_layer = *it;
		net.add(juml::ann::make_SigmoidLayer(n_features, *it));
		it++;
		for (; it != hidden_layers.end(); it++) {
			net.add(juml::ann::make_SigmoidLayer(previous_layer, *it));
			previous_layer = *it;
		}
		net.add(juml::ann::make_SigmoidLayer(previous_layer, n_classes));
		struct stat buffer;
		if (stat(networkFilePath.c_str(), &buffer) == 0) {
			// File exists
			net.load(networkFilePath);
			cout << "ANN loaded from file " << networkFilePath << endl;
			// TODO Check that ANN loaded from file matches specification from command line!
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

	af::array label_array = label.data().as(s32);
	//label_array = label_array(af::span, af::seq(0, label_array.dims(1) - 1, dataset_stepsize));
	//label_array = label_array(af::span, af::seq(0, 8000/mpi_size - 1));

	int min_label = af::min<int>(label_array);
	MPI_Allreduce(MPI_IN_PLACE, &min_label, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
	int max_label = af::max<int>(label_array);
	MPI_Allreduce(MPI_IN_PLACE, &max_label, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

	if (data_array.numdims() > 2) {
		printf("Data Array has more than 2 dimensions. Transforming to 2 dimensions, keeping the first dimension and collapsing the others.\n");
		data_array = af::moddims(data_array, n_features, data_array.dims(data_array.numdims() - 1));
	}

	if (min_label == 0) {
		printf("Found 0-based labels\n");
		if (max_label != n_classes - 1) {
			printf("The biggest label is %d and not %d. Is your number of classes correct?\n", max_label, n_classes - 1);
			MPI_Finalize();
			exit(1);
		}
	} else if (min_label == 1) {
		printf("Found 1-based labels\n");
		label_array -= 1;
		if (max_label != n_classes) {
			printf("The biggest label is %d and not %d. Is your number of classes correct?\n", max_label, n_classes);
			MPI_Finalize();
			exit(1);
		}
	} else {
		printf("Labels need to be 1 or 0 based!\n");
		MPI_Finalize();
		exit(1);
	}

	const int N = data_array.dims(1);
	int globalN;
	MPI_Allreduce(&N, &globalN, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	af::array target = af::constant(0, n_classes, N, s32);
	target(af::range(N).T() *  n_classes + label_array)= 1;

	double time_splitted_data = MPI_Wtime();
	double time_train_batch_test = 0;
	double time_train_sync = 0;

	if (N < batchsize) {
		cout << "batchsize is bigger than available samples. reducing batchsize to all samples" << endl;
		batchsize = N;
	}

	int nbatches = N/batchsize;
	cout << "N: " << N << " n_batches: " << nbatches << endl
		<< "batchsize: " << batchsize <<endl;
	if (mpi_rank == 0) {
		printf("%5s %10s %10s %10s\n", "Epoch", "Error", "Last Error", "Accuracy");
	}

	af::array shuffled_idx, sorted_randomizer;

	for (int epoch = 0; epoch < max_epochs; epoch++) {
		if (shuffle_samples) {
			//Generate shuffled array of indexes
			af::sort(sorted_randomizer, shuffled_idx, af::randu(N));
		}
		float error = 0;
		float lasterror;
		for (int batch = 0, batchnum = 0; batch < N; batchnum++) {
			int thisbatchsize = batchsize + (N % batchsize) / nbatches + (batchnum < (N % batchsize) % nbatches ? 1 : 0);
			int last = std::min(static_cast<int>(data_array.dims(1)) - 1, batch + thisbatchsize - 1);
			af::array batchsamples, batchtarget;
			if (shuffle_samples) {
				af::array batchidx = shuffled_idx(af::seq(batch, last));
				batchsamples = data_array(af::span, batchidx);
				batchtarget = target(af::span, batchidx);
			} else {
				batchsamples = data_array(af::span, af::seq(batch, last));
				batchtarget = target(af::span, af::seq(batch, last));
			}
			//Only sync with other processes if sync_after_batch_update is set
			lasterror = net.fitBatch(batchsamples, batchtarget, LEARNINGRATE, sync_after_batch_update ? MPI_COMM_NULL : MPI_COMM_SELF);
			error += lasterror;
			batch += thisbatchsize;
		}
		double time_buf = MPI_Wtime();

		if (!sync_after_batch_update) {
			net.sync();
		}

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
