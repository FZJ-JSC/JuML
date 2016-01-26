#ifndef JUML_ANNLAYERS_H_
#define JUML_ANNLAYERS_H_
#include<arrayfire.h>
namespace juml {
	namespace ann {
		class Layer {
			protected:
				af::array weights;
				af::array weights_update;
				af::array bias;
				af::array bias_update;
				af::array lastOutput;
				int update_count = 0;
			public:
				Layer(int input_size, int node_count) :
					weights(af::randu(input_size, node_count)),
					weights_update(af::constant(0, input_size, node_count)),
					bias(af::randu(node_count)),
					bias_update(af::constant(0, node_count)),
					lastOutput(node_count) {}

				void updateWeights(float learningrate) {
					if (update_count == 0) return;
					this->weights_update /= this->update_count;
					this->bias_update /= this->update_count;
					this->weights -= learningrate * this->weights_update;
				        this->bias -= learningrate * this->bias_update;

					this->weights_update(af::span) = 0;
					this->bias_update(af::span) = 0;
					this->update_count = 0;
				}

				inline af::array& getWeightUpdates() {
					return this->weights_update;
				}

				inline af::array& getBiasUpdates() {
					return this->bias_update;
				}

				virtual const af::array& forward(const af::array& input) = 0;

				/**
				 * Update weights_update by backpropagation with the input, that produced the last output, the delta value for the expected output and the learningrate.
				 */
				virtual const af::array& backwards(
						const af::array& input,
						const af::array& delta) = 0;

				inline const af::array& getLastOutput() const {
					return lastOutput;
				}

				virtual ~Layer() {};
		};

		//TODO implement a template-'FunctionLayer' with template-spezialisations for different activation functions?

		class SigmoidLayer : Layer {
			SigmoidLayer(int input_size, int node_count) : Layer(input_size, node_count) {}

			inline af::array sigmoid_deriv(af::array &out) {
				return out * (1 - out);
			}

			const af::array& forward(const af::array& input) override {
				af::array sumOfWeightedInputs = af::matmul(this->weights, input) + this->bias;
				this->lastOutput = af::sigmoid(sumOfWeightedInputs);
				return lastOutput;
			}

			const af::array& backwards(
					const af::array& input,
					const af::array& lastDelta) override {
				af::array d = lastDelta * sigmoid_deriv(lastOutput);
				this->weights_update += matmul(input, d);
				this->bias_update += d;
				this->update_count += 1;
			}
		};
	}
}

#endif
