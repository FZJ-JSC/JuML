#ifndef CLASSNORMALIZER_H
#define CLASSNORMALIZER_H

#include <armadillo>
#include <exception>
#include <map>
#include <sstream>
#include <stdint.h>

namespace juml {
namespace preprocessing {

    class ClassNormalizer {
    protected:
        arma::ivec class_labels_;
        std::map<int32_t, int32_t> class_mapping_;
    public:
        void index(const arma::ivec& y);

        inline int32_t n_classes() const {
            return this->class_labels_.n_elem;
        }

        int32_t transform(int32_t class_label) const;

        inline int32_t invert(int32_t transformed_label) const {
            return this->class_labels_(transformed_label);
        }

        inline const arma::ivec& classes() const {
            return this->class_labels_;
        }
    };

} // namespace preprocessing
} // namespace juml

#endif // CLASSNORMALIZER_H
