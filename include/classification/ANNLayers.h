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
				const int input_count;
				const int node_count;
				Layer(int input_count_, int node_count_) :
					weights(af::randu(input_count_, node_count_)),
					weights_update(af::constant(0, input_count_, node_count_)),
					bias(af::randu(node_count_)),
					bias_update(af::constant(0, node_count_)),
					lastOutput(node_count_),
		       			input_count(input_count_), node_count(node_count_) {}

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
				 * Result will be written into lastOutput!
				 */
				virtual const af::array& backwards(
						const af::array& input,
						const af::array& delta) = 0;

				/**
				 * Return the last output of the layer or the last delta value, depending on if forward or backwards was caled last.
				 */
				inline const af::array& getLastOutput() const {
					return lastOutput;
				}

				virtual ~Layer() {};
		};

		//TODO implement a template-'FunctionLayer' with template-spezialisations for different activation functions?

		class SigmoidLayer : public Layer {
			public: 
			SigmoidLayer(int input_size, int node_count) : Layer(input_size, node_count) {}

			inline af::array sigmoid_deriv(af::array &out) {
				return out * (1 - out);
			}

			const af::array& forward(const af::array& input) override {
				// transpose(IxN) * I + N = NxI * I + N = N
				af::array sumOfWeightedInputs = af::matmulTN(this->weights, input) + this->bias;
				this->lastOutput = af::sigmoid(sumOfWeightedInputs);
				return lastOutput;
			}

			const af::array& backwards(
					const af::array& input /* column with input_count rows */,
					const af::array& lastDelta /* column with node_count rows */) override {
				// scalar_mult(N, N) = N 
				af::array d = lastDelta * sigmoid_deriv(this->lastOutput);
				// I * transpose(d) = (Ix1) * transpose(Nx1) = (Ix1)*(1xN) = IxN
				this->weights_update += matmulNT(input, d);
				this->bias_update += d;
				this->update_count += 1;
				// IxN * N = I;
				this->lastOutput = af::matmul(this->weights, lastDelta);
				return this->lastOutput;
			}
		};
	}
}

#endif
