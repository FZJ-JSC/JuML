#ifndef JUML_ANNLAYERS_H_
#define JUML_ANNLAYERS_H_
#include<arrayfire.h>
#include<iostream>
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
					weights(
						af::randu(input_count_, node_count_)
						* (1.0f/input_count_) - (0.5f/input_count_)
					),
					weights_update(af::constant(0, input_count_, node_count_)),
					bias(
						af::randu(node_count_)
						* (1.0f/input_count_) - (0.5f/input_count_)
					),
					bias_update(af::constant(0, node_count_)),
					lastOutput(node_count_),
		       			input_count(input_count_), node_count(node_count_) {}

				void updateWeights(float learningrate) {
					if (update_count == 0) return;
					//TODO: Sync weights_update, bias_update and update_count with other processes
					this->weights_update /= this->update_count;
					this->bias_update /= this->update_count;

					this->weights -= learningrate * this->weights_update;
				        this->bias -= learningrate * this->bias_update;

					this->weights_update(af::span, af::span) = 0;
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
				 * Return the last output of the layer or the last delta value, depending on if forward or backwards was called last.
				 */
				inline const af::array& getLastOutput() const {
					return lastOutput;
				}

				virtual ~Layer() {
#ifndef NDEBUG
					std::cout << "Layer Deconstructor called" << std::endl;
#endif
				};
		};

		enum class Activation { Sigmoid, Linear };

		template<Activation T>
		inline af::array activation(const af::array& in) {
			throw std::runtime_error("Not implemented");
		}

		template<Activation T>
		inline af::array activation_deriv(const af::array& out) {
			throw std::runtime_error("Not implemented");
		}
//Linear:
		template<>
		inline af::array activation<Activation::Linear>(const af::array& in) {
			return in;
		}

		template<>
		inline af::array activation_deriv<Activation::Linear>(const af::array& out) {
			return af::constant(1.0f, out.dims(0), out.dims(1));
		}

//Sigmoid:
		template<>
		inline af::array activation<Activation::Sigmoid>(const af::array& in) {
			return af::sigmoid(in);
		}
		template<>
		inline af::array activation_deriv<Activation::Sigmoid>(const af::array &out) {
			return out * (1 - out);
		}

		//TODO implement a template-'FunctionLayer' with template-spezialisations for different activation functions?
		template<Activation T>
		class FunctionLayer: public Layer {
			public: 
			FunctionLayer(int input_size, int node_count) : Layer(input_size, node_count) {}

			const af::array& forward(const af::array& input) override {
				// matmul(transpose(IxN), (Ixb))  =
				// matmul(         (NxI), (Ixb))  = (Nxb)
				af::array sumOfWeightedInputs = af::matmulTN(this->weights, input);
				// Nxb += tile(Nx1, 1, b) = Nxb
				sumOfWeightedInputs += af::tile(this->bias, 1, sumOfWeightedInputs.dims(1));
				this->lastOutput = activation<T>(sumOfWeightedInputs);
				return lastOutput;
			}

			const af::array& backwards(
					const af::array& input /* column with input_count rows and batchsize columns*/,
					const af::array& lastDelta /* column with node_count rows and batchsize columns */) override {
				// scalar_mult(Nxb, Nxb) = Nxb
				af::array d = lastDelta * activation_deriv<T>(this->lastOutput);
				// matmul(Ixb, transpose(Nxb)) = matmul(Ixb, bxN) = (Ixb)*(bxN) = IxN
				this->weights_update += matmulNT(input, d);
				// (Nx1) += sum(Nxb, 1) = Nx1
				this->bias_update += af::sum(d, 1);
				this->update_count += input.dims(1);
				// matmul(IxN, Nxb) = Ixb;
				this->lastOutput = af::matmul(this->weights, lastDelta);
				return this->lastOutput;
			}
		};
	}
}

#endif
