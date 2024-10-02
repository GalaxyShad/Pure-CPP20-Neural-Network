#include "neural_network.h"

NeuralNetwork::NeuralNetwork(NeuralNetworkModel &model) : NeuralNetwork(model.layers_sizes_vector()) {
    weight_matrix_ = model.weights();
}

NeuralNetwork::NeuralNetwork(const size_t *layers_size_list, size_t layers_count) {
    for (int i = 0; i < layers_count; i++) {
        if (i != 0) weight_matrix_.emplace_back(layers_size_list[i], layers_size_list[i - 1]);
        layers_.emplace_back(layers_size_list[i]);
    }
}

auto NeuralNetwork::predict_to_output(const f32 *input_data) -> void {
    auto &input_layer = layers_.front();

    for (auto i = 0; i < input_layer.neurons().size(); i++) {
        input_layer.neurons().at(i).value = input_data[i];
    }

    for (auto k = 1; k < layers_.size(); k++) {
        auto &layer = layers_[k];
        auto &weights = weight_matrix_[k - 1];
        auto &prev_layer = layers_[k - 1];

        for (int i = 0; i < layer.neurons().size(); i++) {
            auto &neuron = layer.neurons().at(i);

            f32 sum = 0.f;
            for (auto j = 0; j < prev_layer.neurons().size(); j++) {
                auto w = weights.weight_between(i, j);
                auto pn = prev_layer.neurons().at(j).value;

                sum += w * pn;
            }

            neuron.value = sigmoid(sum);
        }
    }
}

auto NeuralNetwork::predict(const std::vector<f32> &input) -> std::vector<f32> {
    predict_to_output(input.data());

    std::vector<f32> out(layers_.back().neurons().size());
    std::transform(layers_.back().neurons().begin(), layers_.back().neurons().end(), out.begin(),
                   [](const auto &neuron) { return neuron.value; });

    return out;
}

auto NeuralNetwork::train(const NeuralNetwork::TrainingParams &params,
                          ITrainingObserver *observer) -> NeuralNetwork::TrainingResult {
    i32 epoch = 0;
    f32 err = 0.f;

    auto time_start = std::chrono::high_resolution_clock::now();

    for (; epoch < params.max_epochs; epoch++) {
        auto is_error = false;

        for (int i = 0; i < params.train_data_count; i++) {
            predict_to_output(params.input_data + i * layers_.front().neurons().size());

            auto expected_output = params.output_data + (i * layers_.back().neurons().size());
            auto input = params.input_data + (i * layers_.front().neurons().size());

            err = compute_error(expected_output);

            if (err > params.min_err) {
                back_propagate(expected_output, params.back_propagation_speed);
                predict_to_output(input);
                is_error = true;
            }

            if (observer != nullptr) {
                observer->on_epoch({
                                           .current_error = err,
                                           .current_epoch = epoch,
                                           .nn_weights = weight_matrix_,
                                           .nn_layers = layers_,
                                           .data_set_index = i
                                   });
            }
        }

        if (!is_error) break;
    }

    auto time_end = std::chrono::high_resolution_clock::now() - time_start;

    return TrainingResult{
            .model = NeuralNetworkModel(weight_matrix_),
            .epoch_count = epoch,
            .error = err,
            .time = std::chrono::duration_cast<std::chrono::milliseconds>(time_end).count()
    };
}

auto NeuralNetwork::compute_error(const f32 *expected_output_list) -> f32 {
    auto sum = 0.f;

    for (int i = 0; i < layers_.back().neurons().size(); i++) {
        sum += abs(expected_output_list[i] - layers_.back().neurons().at(i).value);
    }

    return sum / 2.f;
}

auto NeuralNetwork::compute_beta(i32 layer_index, i32 neuron_index, const f32 *expected_output_list) -> f32 {
    auto &neuron = layers_[layer_index].neurons().at(neuron_index);

    auto y = neuron.value;
    auto beta = y * (1.f - y);

    if (layer_index == layers_.size() - 1) {
        beta *= expected_output_list[neuron_index] - y;
    } else {
        auto &next_layer = layers_[layer_index + 1];
        auto &next_weights = weight_matrix_[layer_index];

        auto sum = 0.f;
        for (int j = 0; j < next_layer.neurons().size(); j++) {
            sum += next_layer.neurons().at(j).beta * next_weights.weight_between(j, neuron_index);
        }
        beta *= sum;
    }

    return beta;
}

auto NeuralNetwork::back_propagate(const f32 *&expected_outputs_list, f32 speed) -> void {
    for (auto k = layers_.size() - 1; k >= 1; k--) {
        auto &weights = weight_matrix_[k - 1];

        for (int i = 0; i < weights.out_count(); i++) {
            auto beta = compute_beta(k, i, expected_outputs_list);

            auto &neuron = layers_[k].neurons().at(i);

            neuron.beta = beta;

            for (int j = 0; j < weights.in_count(); j++) {
                auto delta = speed * beta * layers_[k - 1].neurons().at(j).value;

                auto new_weight = weights.weight_between(i, j) + delta;

                weights.set_weight(i, j, new_weight);
            }
        }
    }
}
