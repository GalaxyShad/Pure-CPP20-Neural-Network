#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "neural_network_model.h"
#include "weight_matrix.h"
#include "activators.h"

struct Neuron {
    f32 value, beta;
};

class Layer {
public:
    explicit Layer(size_t neuron_count) : neurons_(neuron_count) {}

    auto &neurons() { return neurons_; }

private:
    std::vector<Neuron> neurons_;
};

struct TrainingObserverData {
    f32 current_error;
    i32 current_epoch;
    std::vector<WeightMatrix> &nn_weights;
    std::vector<Layer> &nn_layers;
    i32 data_set_index;
};

struct ITrainingObserver {
    virtual auto on_epoch(const TrainingObserverData &data) -> void = 0;
};

class NeuralNetwork {
public:
    explicit NeuralNetwork(const size_t *layers_size_list, size_t layers_count);

    explicit NeuralNetwork(const std::vector<size_t> &layers_size) : NeuralNetwork(layers_size.data(),
                                                                                   layers_size.size()) {}

    explicit NeuralNetwork(NeuralNetworkModel &model);

public:
    auto output(i32 i) -> f32 {
        return layers_.back().neurons().at(i).value;
    }

    auto predict_to_output(const f32 *input_data) -> void;

    auto predict(const std::vector<f32> &input) -> std::vector<f32>;

    struct TrainingParams {
        size_t train_data_count = 0;
        const f32 *input_data = nullptr;
        const f32 *output_data = nullptr;
        i32 max_epochs = 2'000'000;
        f32 min_err = 0.025f;
        f32 back_propagation_speed = 0.1f;
    };

    struct TrainingResult {
        NeuralNetworkModel model;
        i32 epoch_count;
        f32 error;
        i64 time;
    };

    auto train(const TrainingParams &params, ITrainingObserver *observer = nullptr) -> TrainingResult;

    auto compute_error(const f32 *expected_output_list) -> f32;


private:
    auto compute_beta(i32 layer_index, i32 neuron_index, const f32 *expected_output_list) -> f32;

    auto back_propagate(const f32 *&expected_outputs_list, f32 speed = 0.1f) -> void;

private:
    std::vector<Layer> layers_;
    std::vector<WeightMatrix> weight_matrix_;
};

#endif //NEURAL_NETWORK_H
