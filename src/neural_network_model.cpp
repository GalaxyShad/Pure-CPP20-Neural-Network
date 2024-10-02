#include "neural_network.h"

auto NeuralNetworkModel::layer_size(i32 layer_index) {
    return (layer_index == 0)
           ? weights_[layer_index].in_count()
           : weights_[layer_index - 1].out_count();
}

auto NeuralNetworkModel::layers_sizes_vector() -> std::vector<size_t> {
    auto n = layers_count();

    std::vector<size_t> res(n);

    for (int i = 0; i < n; i++) {
        res[i] = layer_size(i);
    }

    return res;
}

auto NeuralNetworkModel::serialize(NeuralNetworkModel &model) -> std::vector<u8> {


    // Calculate the total size needed for the buffer
    u64 len = model.weights().size();
    u64 total_size = sizeof(len);

    for (const auto &w: model.weights()) {
        total_size += sizeof(i32) * 2; // For out_count and in_count
        total_size += sizeof(f32) * w.out_count() * w.in_count(); // For weights
    }

    std::vector<u8> buff(total_size);

    auto *ptr = buff.data();

    std::memcpy(ptr, &len, sizeof(len)), ptr += sizeof(len);

    for (auto &w: model.weights()) {
        i32 matrix_out = w.out_count();
        i32 matrix_in = w.in_count();

        std::memcpy(ptr, &matrix_out, sizeof(matrix_out)), ptr += sizeof(matrix_out);
        std::memcpy(ptr, &matrix_in, sizeof(matrix_in)), ptr += sizeof(matrix_in);

        for (auto i = 0; i < w.out_count(); i++) {
            for (auto j = 0; j < w.in_count(); j++) {
                f32 weight = w.weight_between(i, j);
                std::memcpy(ptr, &weight, sizeof(weight)), ptr += sizeof(weight);
            }
        }
    }

    return buff;
}

auto NeuralNetworkModel::deserialize(std::vector<u8> &buff) -> NeuralNetworkModel {
    auto *ptr = buff.data();

    std::vector<WeightMatrix> weights;

    size_t len;
    std::memcpy(&len, ptr, sizeof(len)), ptr += sizeof(len);

    for (auto k = 0; k < len; k++) {
        i32 matrix_out, matrix_in;
        std::memcpy(&matrix_out, ptr, sizeof(matrix_out)), ptr += sizeof(matrix_out);
        std::memcpy(&matrix_in, ptr, sizeof(matrix_in)), ptr += sizeof(matrix_in);

        weights.emplace_back(matrix_out, matrix_in);

        auto &w = weights.back();

        for (auto i = 0; i < w.out_count(); i++) {
            for (auto j = 0; j < w.in_count(); j++) {
                f32 weight;
                std::memcpy(&weight, ptr, sizeof(weight)), ptr += sizeof(weight);
                w.set_weight(i, j, weight);
            }
        }
    }

    return NeuralNetworkModel(weights);
}


