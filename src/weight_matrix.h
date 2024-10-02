#ifndef WEIGHT_MATRIX_H
#define WEIGHT_MATRIX_H

#include <vector>
#include "shorttypes.h"

class WeightMatrix {
public:
    explicit WeightMatrix(i32 out_count, i32 in_count) : in_count_(in_count), out_count_(out_count),
                                                         weights_(in_count * out_count) {
        for (auto &w: weights_) {
            w = (rand() / (float) RAND_MAX) * 0.5f;
        }
    }

public:
    inline auto set_weight(i32 out, i32 in, f32 w) { weights_[out * in_count_ + in] = w; }

    inline auto weight_between(i32 out, i32 in) { return weights_[out * in_count_ + in]; }

    auto out_count() const { return out_count_; }

    auto in_count() const { return in_count_; }

private:
    std::vector<f32> weights_;
    i32 in_count_;
    i32 out_count_;
};

#endif //WEIGHT_MATRIX_H
