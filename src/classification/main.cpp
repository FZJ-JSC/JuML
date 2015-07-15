#include "classification/GaussianNaiveBayes.h"
#include <armadillo>
#include <iostream>

int main() {
    arma::fmat daten;
    daten.load("iris_daten.txt", arma::raw_ascii);
    arma::ivec labels;
    labels.load("iris_labels.txt", arma::raw_ascii);

    juml::classification::GaussianNaiveBayes nb;
    nb.fit(daten,labels);

    std::cout.precision(10);
    std::cout <<  nb.score(daten,labels) << std::endl;
    return 0;
}
