#ifndef NEURAL_NETWORK_MODEL_H
#define NEURAL_NETWORK_MODEL_H

#include <vector>
#include "shorttypes.h"
#include "weight_matrix.h"

class NeuralNetworkModel {
public:
    explicit NeuralNetworkModel(std::vector<WeightMatrix> weights) : weights_(std::move(weights)) {}

public:
    auto &weights() { return weights_; }

    auto layer_size(i32 layer_index);

    auto layers_count() { return weights_.size() + 1; }

    auto layers_sizes_vector() -> std::vector<size_t>;

    static auto serialize(NeuralNetworkModel &model) -> std::vector<u8>;

    static auto deserialize(std::vector<u8> &buff) -> NeuralNetworkModel;

private:
    std::vector<WeightMatrix> weights_;
};

#endif //NEURAL_NETWORK_MODEL_H
