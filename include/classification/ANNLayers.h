#ifndef JUML_ANNLAYERS_H_
#define JUML_ANNLAYERS_H_
#include<arrayfire.h>
#include<iostream>
#include<mpi.h>
#include "core/MPI.h"
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

				virtual void applyWeightUpdate(float learningrate, MPI_Comm comm) {
					this->weights -= learningrate * this->weights_update;
				        this->bias -= learningrate * this->bias_update;
				}
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

				Layer(af::array weights_, af::array bias_) :
					weights(weights_), bias(bias_),
					weights_update(af::constant(0, weights_.dims(0), weights_.dims(1))),
					bias_update(af::constant(0.0, bias_.dims(0))),
					lastOutput(weights_.dims(1)),
					input_count(weights_.dims(0)), node_count(weights_.dims(1))
				{
					if (weights.dims(1) != bias.dims(0)) {
						throw std::runtime_error("Node count in weight matrix and bias vector differ");
					}
				}

				void updateWeights(float learningrate, MPI_Comm comm) {
					MPI_Allreduce(MPI_IN_PLACE, &this->update_count, 1, MPI_INT, MPI_SUM, comm);
					if (update_count == 0) return;

					mpi::allreduce_inplace(this->weights_update, MPI_SUM, comm);
					mpi::allreduce_inplace(this->bias_update, MPI_SUM, comm);

					this->weights_update /= this->update_count;
					this->bias_update /= this->update_count;

					applyWeightUpdate(learningrate, comm);

					this->weights_update(af::span, af::span) = 0;
					this->bias_update(af::span) = 0;
					this->update_count = 0;
				}

				inline af::array& getWeights() {
					return this->weights;
				}

				inline af::array& getBias() {
					return this->bias;
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

		typedef std::shared_ptr<Layer> LayerPtr;

		enum class Activation { Sigmoid, TanH, Linear };

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
//TanH
		template<>
		inline af::array activation<Activation::TanH>(const af::array& in) {
			return af::tanh(in);
		}

		template<>
		inline af::array activation_deriv<Activation::TanH>(const af::array &out) {
			return (1 - out * out);
		}

		//TODO implement a template-'FunctionLayer' with template-spezialisations for different activation functions?
		template<Activation T>
		class FunctionLayer: public Layer {
			public: 
			FunctionLayer(int input_size, int node_count) : Layer(input_size, node_count) {}
			FunctionLayer(af::array weights, af::array bias) : Layer(weights, bias) {}

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
					const af::array& error /* column with node_count rows and batchsize columns */) override {
				// scalar_mult(Nxb, Nxb) = Nxb
				af::array delta = error * activation_deriv<T>(this->lastOutput);
				// matmul(Ixb, transpose(Nxb)) = matmul(Ixb, bxN) = (Ixb)*(bxN) = IxN
				this->weights_update += matmulNT(input, delta);
				// (Nx1) += sum(Nxb, 1) = Nx1
				this->bias_update += af::sum(delta, 1);
				this->update_count += input.dims(1);
				// matmul(IxN, Nxb) = Ixb;
				this->lastOutput = af::matmul(this->weights, delta);
				return this->lastOutput;
			}
		};

		template<Activation T>
		class MomentumFunctionLayer: public FunctionLayer<T> {
			public:
			MomentumFunctionLayer(int input_size, int node_count, float momentum) :
				FunctionLayer<T>(input_size, node_count),
				previous_weight_update(af::constant(0.0, input_size, node_count)),
				previous_bias_update(af::constant(0.0, node_count)),
				momentum_(momentum) {
			}

			protected:
			af::array previous_weight_update;
			af::array previous_bias_update;
			float momentum_;

			void applyWeightUpdate(float learningrate, MPI_Comm comm) override {
				this->weights -= learningrate * (
					       (1 - this->momentum_) * this->weights_update +
					       this->momentum_ * this->previous_weight_update);
				this->bias -= learningrate * (
						(1 - this->momentum_) * this->bias_update +
						this->momentum_ * this->previous_bias_update);

				this->previous_weight_update = this->weights_update;
				this->previous_bias_update = this->bias_update;
			}
		};
	}
}

#endif
