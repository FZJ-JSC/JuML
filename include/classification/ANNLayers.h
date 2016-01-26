#ifndef JUML_ANNLAYERS_H_
#define JUML_ANNLAYERS_H_
#include<arrayfire.h>
namespace juml {
	namespace ann {
		class Layer {
			protected:
				af::array lastOutput;
				af::array weights;
				af::array weights_update;
			public:
				Layer(int input_size, int node_count) :
					weights(af::randu(input_size + 1, node_count)),
					weights_update(af::constant(0, input_size + 1, node_count)),
					lastOutput(node_count)
				{

				}
				virtual const af::array& forward(const af::array& input) const = 0;
				/**
				 * Update weights_update by backpropagation with the input, that produced the last output, the delta value for the expected output and the learningrate.
				 */
				virtual const af::array& backwards(
						const af::array& input,
						const af::array& delta,
						float learningrate
						) = 0;
				inline const af::array& getLastOutput() const {
					return lastOutput;
				}
				virtual ~Layer() {};
		};

		//TODO implement a template-'FunctionLayer' with template-spezialisations for different activation functions?

		class SigmoidLayer : Layer {
			//TODO implement SigmoidLayer
		};
	}
}

#endif
