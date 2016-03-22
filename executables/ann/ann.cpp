#include<tclap/CmdLine.h>

int main(int argc, char *argv[]) {
	try {
		TCLAP::CmdLine cmd("Command to train and use an Artificial Neural Network", ' ', "0.1-alpha");

		std::vector<char> activationCharsVector= {'s', 't', 'l'};
		TCLAP::ValuesConstraint<char> activationChars(activationCharsVector);
		TCLAP::ValueArg<char> hiddenActivation("a", "activation",
				"Activation function used in Hidden Layers. Sigmoid, TanH or Linear", false, 's', &activationChars, cmd);
		TCLAP::ValueArg<char> outputActivation("o", "output-layer",
				"Activation function used in output layer.", false, 's', &activationChars, cmd);
		TCLAP::MultiArg<int> layers("l", "layer",
				"Add a Layer to the ANN.", true,  "Number of Nodes", cmd);


		cmd.parse(argc, argv);

	} catch (TCLAP::ArgException &e) {
		std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
	}
}
