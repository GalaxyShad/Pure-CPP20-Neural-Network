#include <iostream>
#include <vector>

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

    inline auto out_count() const { return out_count_; }

    inline auto in_count() const { return in_count_; }

private:
    std::vector<f32> weights_;
    i32 in_count_;
    i32 out_count_;
};


class Layer {
public:
    explicit Layer(size_t neuron_count) : neurons_(neuron_count) {}

    auto& neuron(i32 index) { return neurons_[index]; }

    auto size() const { return neurons_.size(); }

private:
    std::vector<Neuron> neurons_;
};

class NeuralNetwork {
public:
    explicit NeuralNetwork(const size_t *layers_size_list, size_t layers_count) {
        for (int i = 0; i < layers_count; i++) {
            if (i != 0)
                weight_matrix_.emplace_back(layers_size_list[i], layers_size_list[i - 1]);
            layers_.emplace_back(layers_size_list[i]);
        }
    }

public:
    auto output(i32 i) -> f32 {
        return layers_[layers_.size() - 1].neuron(i).value;
    }

    auto predict(const f32 *input_data) {
        auto &input_layer = layers_[0];

        for (int i = 0; i < input_layer.size(); i++) {
            input_layer.neuron(i).value = input_data[i];
        }

        for (int k = 1; k < layers_.size(); k++) {
            auto &layer = layers_[k];
            auto &weights = weight_matrix_[k - 1];
            auto &prev_layer = layers_[k - 1];

            for (int i = 0; i < layer.size(); i++) {
                auto &neuron = layer.neuron(i);

                auto sum = 0.f;
                for (int j = 0; j < prev_layer.size(); j++) {
                    auto w = weights.weight_between(i, j);
                    auto pn = prev_layer.neuron(j).value;

                    sum += w * pn;
                }

                neuron.value = activate(sum);
            }
        }
    }

    auto train(size_t train_data_count, const f32 *input_data, const f32 *output_data) -> void {
        for (i32 epoch = 0; epoch < 200'000; epoch++) {
            auto is_error = false;

            for (int i = 0; i < train_data_count; i++) {
                predict(input_data + i * layers_[0].size());

                auto expected = output_data + i * layers_[layers_.size() - 1].size();

                auto err = compute_error(expected);

                if (err > 0.025f) {

                    back_propagate(expected);
                    predict(input_data + i * layers_[0].size());
                    is_error = true;
                }
            }

            if (!is_error) break;
        }
    }

    auto compute_error(const f32 *expected_output_list) -> f32 {
        auto sum = 0.f;

        auto &endLayer = layers_[layers_.size() - 1];

        for (int i = 0; i < endLayer.size(); i++) {
            sum += abs(expected_output_list[i] - endLayer.neuron(i).value);
        }

        return sum / 2.f;
    }



private:
    auto compute_beta(i32 layer_index, i32 neuron_index, const f32 *expected_output_list) -> f32 {
        auto &neuron = layers_[layer_index].neuron(neuron_index);

        auto y = neuron.value;
        auto beta = y * (1.f - y);

        if (layer_index == layers_.size() - 1) {
            beta *= expected_output_list[neuron_index] - y;
        } else {
            auto &next_layer = layers_[layer_index + 1];
            auto &next_weights = weight_matrix_[layer_index];

            auto sum = 0.f;
            for (int j = 0; j < next_layer.size(); j++) {
                sum += next_layer.neuron(j).beta * next_weights.weight_between(j, neuron_index);
            }
            beta *= sum;
        }

        return beta;
    }

    auto back_propagate(const f32 *&expected_outputs_list, f32 alpha = 0.1f) -> void {
        for (int k = layers_.size() - 1; k >= 1; k--) {
            auto &weights = weight_matrix_[k - 1];

            for (int i = 0; i < weights.out_count(); i++) {
                auto beta = compute_beta(k, i, expected_outputs_list);

                auto &neuron = layers_[k].neuron(i);

                neuron.beta = beta;

                for (int j = 0; j < weights.in_count(); j++) {
                    auto delta = alpha * beta * layers_[k - 1].neuron(j).value;

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

    nn.predict(new f32[]{1.f, 0.25f});

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
    size_t layers[] = {2, 16,  1};

    NeuralNetwork nn(layers, sizeof(layers) / sizeof(layers[0]));

    f32 train_data_in[] = {
            0.f, 0.f,
            0.f, 1.f,
            1.f, 0.f,
            1.f, 1.f
    };

    f32 train_data_out[] = {
            0, 1.f, 1.f, 0.f
    };

    nn.train(4, train_data_in, train_data_out);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            float input[] = {(f32) i, (f32) j};
            nn.predict(input);
            std::cout << i << " xor " << j << " = " << nn.output(0) << std::endl;
        }
    }

    return 0;
}

