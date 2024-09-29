#include <iostream>
#include <vector>
#include <numeric>

using f32 = float;
using i32 = int;

auto activate(f32 x) -> f32 {
    return 1.f / (1.f + exp(-x));
}

struct Neuron {
    f32 value, beta;
};

class WeightMatrix {
public:
    explicit WeightMatrix(i32 out_count, i32 in_count) : in_count_(in_count), out_count_(out_count),
                                                         weights_(in_count * out_count) {
        for (auto &w: weights_) {
            w = (random() / (float) RAND_MAX) * 0.5f;
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


class Layer {
public:
    explicit Layer(size_t neuron_count) : neurons_(neuron_count) {}

    auto &neurons() { return neurons_; }

private:
    std::vector<Neuron> neurons_;
};

class NeuralNetwork {
public:
    explicit NeuralNetwork(const size_t *layers_size_list, size_t layers_count) {
        for (int i = 0; i < layers_count; i++) {
            if (i != 0) weight_matrix_.emplace_back(layers_size_list[i], layers_size_list[i - 1]);
            layers_.emplace_back(layers_size_list[i]);
        }
    }

    explicit NeuralNetwork(const std::vector<size_t>& layers_size) : NeuralNetwork(layers_size.data(), layers_size.size()) {}

public:
    auto output(i32 i) -> f32 {
        return layers_.back().neurons().at(i).value;
    }

    auto predict_to_output(const f32 *input_data) {
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

                neuron.value = activate(sum);
            }
        }
    }

    auto predict(const std::vector<f32>& input) {
        predict_to_output(input.data());

        std::vector<f32> out(layers_.back().neurons().size());
        std::transform(layers_.back().neurons().begin(), layers_.back().neurons().end(), out.begin(),
                       [](const auto& neuron) { return neuron.value; });

        return out;
    }

    struct TrainingParams {
        size_t train_data_count = 0;
        const f32 *input_data = nullptr;
        const f32 *output_data = nullptr;
        i32 max_epochs = 2'000'000;
        f32 min_err = 0.025f;
        f32 back_propagation_speed = 0.1f;
    };

    auto train(const TrainingParams& params) -> void {
        for (i32 epoch = 0; epoch < params.max_epochs; epoch++) {
            auto is_error = false;

            for (int i = 0; i < params.train_data_count; i++) {
                predict_to_output(params.input_data + i * layers_.front().neurons().size());

                auto expected_output = params.output_data + (i * layers_.back().neurons().size());
                auto input = params.input_data + (i * layers_.front().neurons().size());

                auto err = compute_error(expected_output);

                if (err > params.min_err) {
                    back_propagate(expected_output, params.back_propagation_speed);
                    predict_to_output(input);
                    is_error = true;
                }
            }

            if (!is_error) break;
        }
    }

    auto compute_error(const f32 *expected_output_list) -> f32 {
        auto sum = 0.f;

        for (int i = 0; i < layers_.back().neurons().size(); i++) {
            sum += abs(expected_output_list[i] - layers_.back().neurons().at(i).value);
        }

        return sum / 2.f;
    }


private:
    auto compute_beta(i32 layer_index, i32 neuron_index, const f32 *expected_output_list) -> f32 {
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

    auto back_propagate(const f32 *&expected_outputs_list, f32 speed = 0.1f) -> void {
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

private:
    std::vector<Layer> layers_;
    std::vector<WeightMatrix> weight_matrix_;
};

auto test() -> void {
    size_t layers[] = {2, 2};
    NeuralNetwork nn(layers, sizeof(layers) / sizeof(layers[0]));

    nn.predict_to_output(new f32[]{1.f, 0.25f});

    //               0.4200093864
    //  - ( 1.00f ) ---- (   )
    //  0.391549617 \  /
    //               \/
    //               /\
    //  0.197191462 /  \
    //  - ( 0.25f )  ---- (  )
    //              0.39922002
}

auto main() -> int {
    NeuralNetwork nn({ 2, 16, 1 });

    f32 train_data_in[] = {
            0.f, 0.f,
            0.f, 1.f,
            1.f, 0.f,
            1.f, 1.f
    };

    f32 train_data_out[] = {
            0, 1.f, 1.f, 0.f
    };

    nn.train({
        .train_data_count = 4,
        .input_data = train_data_in,
        .output_data = train_data_out,
    });

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            std::cout << i << " XOR " << j << " = " << nn.predict({ (f32) i, (f32) j }).at(0) << std::endl;
        }
    }

    return 0;
}

