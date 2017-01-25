/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: ANNActivations.h
*
* Description: Header File that describes the activation functions used in ANN Layers.
*
* Maintainer: 
*
* Email: 
*/



#ifndef JUML_ANNACTIVATIONS_H_INCLUDED_
#define JUML_ANNACTIVATIONS_H_INCLUDED_
#include<arrayfire.h>
#include<stdexcept>
namespace juml {
	namespace ann {
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
	} //namespace ann
} //namespace juml

#endif
