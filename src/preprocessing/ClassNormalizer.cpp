#include "preprocessing/ClassNormalizer.h"

namespace juml {
namespace preprocessing {
    void ClassNormalizer::index(const arma::ivec& y) {
        this->class_labels_ = arma::unique(y);
        this->class_mapping_.clear();

        for (uint64_t label = 0; label < this->class_labels_.n_elem; ++label) {
            auto original_class = this->class_labels_(label);
            this->class_mapping_[original_class] = label;
        }
    }

    int32_t ClassNormalizer::transform(int32_t class_label) const {
        auto found = this->class_mapping_.find(class_label);
        if (found == this->class_mapping_.end()) {
            std::stringstream message;
            message << "Class " << class_label << " not found";
            throw std::invalid_argument(message.str().c_str());
        }
        return found->second;
    }
} // preprocessing
} // juml
