#include <iostream>
#include <utility>
#include <vector>
#include <format>
#include <algorithm>
#include <chrono>

#include "shorttypes.h"
#include "utils.h"

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


class Layer {
public:
    explicit Layer(size_t neuron_count) : neurons_(neuron_count) {}

    auto &neurons() { return neurons_; }

private:
    std::vector<Neuron> neurons_;
};

class NeuralNetworkModel {
public:
    explicit NeuralNetworkModel(std::vector<WeightMatrix> weights) : weights_(std::move(weights)) {}

public:
    auto &weights() { return weights_; }

    auto layer_size(i32 layer_index) {
        return (layer_index == 0)
               ? weights_[layer_index].in_count()
               : weights_[layer_index - 1].out_count();
    }

    auto layers_count() { return weights_.size() + 1; }

    auto layers_sizes_vector() {
        auto n = layers_count();

        std::vector<size_t> res(n);

        for (int i = 0; i < n; i++) {
            res[i] = layer_size(i);
        }

        return res;
    }

    static auto serialize(NeuralNetworkModel &model) -> std::vector<u8> {


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

    static auto deserialize(std::vector<u8> &buff) -> auto {
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

private:
    std::vector<WeightMatrix> weights_;
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
    explicit NeuralNetwork(const size_t *layers_size_list, size_t layers_count) {
        for (int i = 0; i < layers_count; i++) {
            if (i != 0) weight_matrix_.emplace_back(layers_size_list[i], layers_size_list[i - 1]);
            layers_.emplace_back(layers_size_list[i]);
        }
    }

    explicit NeuralNetwork(const std::vector<size_t> &layers_size) : NeuralNetwork(layers_size.data(),
                                                                                   layers_size.size()) {}

    explicit NeuralNetwork(NeuralNetworkModel &model) : NeuralNetwork(model.layers_sizes_vector()) {
        weight_matrix_ = model.weights();
    }

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

    auto predict(const std::vector<f32> &input) {
        predict_to_output(input.data());

        std::vector<f32> out(layers_.back().neurons().size());
        std::transform(layers_.back().neurons().begin(), layers_.back().neurons().end(), out.begin(),
                       [](const auto &neuron) { return neuron.value; });

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

    struct TrainingResult {
        NeuralNetworkModel model;
        i32 epoch_count;
        f32 error;
        i64 time;
    };

    auto train(const TrainingParams &params, ITrainingObserver *observer = nullptr) -> TrainingResult {
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

struct StdoutTrainingObserver : public ITrainingObserver {
    auto on_epoch(const TrainingObserverData &data) -> void override {
        auto x1 = data.nn_layers.front().neurons().front().value;
        auto x2 = data.nn_layers.front().neurons().back().value;

        auto z = data.nn_layers.back().neurons().front().value;

        std::cout << std::format("epoch: {}; err: {}; data_set_index: {}; res: {} xor {} = {};{}\n",
                                 data.current_epoch, data.current_error, data.data_set_index, x1, x2, z,
                                 data.data_set_index == 3 ? '\n' : ' ');
    }
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

auto print_training_info_to_stdout(const NeuralNetwork::TrainingResult& training_res) {
    std::cout << "[TRAINING INFO]\n";
    std::cout << std::format("epochs:\t{:>10}\n", training_res.epoch_count);
    std::cout << std::format("error:\t{:>10}\n", training_res.error);
    std::cout << std::format("time millis:\t{:>10}\n", training_res.time);
}

auto nn_xor_train_and_save() {
    NeuralNetwork nn({2, 16, 1});

    f32 train_data_in[] = {
            0.f, 0.f,
            0.f, 1.f,
            1.f, 0.f,
            1.f, 1.f
    };

    f32 train_data_out[] = {
            0, 1.f, 1.f, 0.f
    };


    StdoutTrainingObserver observer;
    auto training_result = nn.train({
                                            .train_data_count = 4,
                                            .input_data = train_data_in,
                                            .output_data = train_data_out,
                                    }, &observer);


    print_training_info_to_stdout(training_result);

    std::cout << "[INFO] Test\n";
    for (int i = 0b00; i <= 0b11; i++) {
        std::vector<f32> in = {(f32) (i >> 1), (f32) (i & 0b1)};
        std::cout << std::format("{} xor {} = {}\n", in[0], in[1], nn.predict(in).at(0));
    }

    auto out_filename = "../xor_model.bin";
    save_binary(out_filename, NeuralNetworkModel::serialize(training_result.model));

    std::cout << std::format("[INFO] XOR model saved as \"{}\"\n", out_filename);

    std::cout << "\n";
}

auto nn_xor_test() {
    auto model_data = load_binary("../xor_model.bin");
    auto model = NeuralNetworkModel::deserialize(model_data);

    NeuralNetwork nn(model);

    for (int i = 0b00; i <= 0b11; i++) {
        std::vector<f32> in = {(f32) (i >> 1), (f32) (i & 0b1)};
        std::cout << std::format("{} xor {} = {}\n", in[0], in[1], nn.predict(in).at(0));
    }

    std::cout << "\n";
}

struct FiguresDataSet {
    size_t images_count;
    size_t image_radius;
    size_t figures_count;

    std::vector<u8> data;
    std::vector<u8> labels;
};

auto generate_square_circle_triangle_images(int each_figure_frames_count) -> auto {
    const int image_radius = 7;

    u8 figures[][image_radius] = {
            {
                    0b0'0'00000'0,
                    0b0'0'01110'0,
                    0b0'0'10001'0,
                    0b0'0'10001'0,
                    0b0'0'10001'0,
                    0b0'0'01110'0,
                    0b0'0'00000'0,

            },
            {
                    0b0'0'00000'0,
                    0b0'0'11111'0,
                    0b0'0'10001'0,
                    0b0'0'10001'0,
                    0b0'0'10001'0,
                    0b0'0'11111'0,
                    0b0'0'00000'0,
            },
            {
                    0b0'0000000,
                    0b0'0001000,
                    0b0'0010100,
                    0b0'0100010,
                    0b0'1000001,
                    0b0'1111111,
                    0b0'0000000,
            }
    };

    size_t figures_count = 3;
    size_t images_count = each_figure_frames_count;
    size_t data_set_count = images_count * figures_count;

    std::vector<u8> data_set;
    std::vector<u8> data_set_labels;

    for (int figure = 0; figure < figures_count; figure++) {
        for (int img = 0; img < images_count; img++) {

            int x[] = {rand() % 7, rand() % 7, rand() % 7, rand() % 7};
            int y[] = {rand() % 7, rand() % 7, rand() % 7, rand() % 7};

            figures[figure][y[0]] ^= (1 << x[0]) & (~(img == 0));
            figures[figure][y[1]] ^= (1 << x[1]) & (~(img == 0));
            figures[figure][y[2]] ^= (1 << x[2]) & (~(img == 0));
            figures[figure][y[3]] ^= (1 << x[3]) & (~(img == 0));

            for (int i = 0; i < image_radius; i++) {
                for (int j = 0; j < image_radius; j++) {
                    data_set.push_back((figures[figure][i] & (0x40 >> j)) != 0);
                }
            }

            figures[figure][y[0]] ^= (1 << x[0]) & (~(img == 0));
            figures[figure][y[1]] ^= (1 << x[1]) & (~(img == 0));
            figures[figure][y[2]] ^= (1 << x[2]) & (~(img == 0));
            figures[figure][y[3]] ^= (1 << x[3]) & (~(img == 0));

            data_set_labels.push_back(figure == 0);
            data_set_labels.push_back(figure == 1);
            data_set_labels.push_back(figure == 2);
        }
    }

    for (int k = 0; k < data_set_count; k++) {
        for (int i = 0; i < image_radius; i++) {
            for (int j = 0; j < image_radius; j++) {
                std::cout << ((data_set[k * 49 + i * 7 + j] != 0) ? '#' : '.');
            }
            std::cout << "\n";
        }

        std::cout << k << "\n";
    }

    return FiguresDataSet{
            .images_count = data_set_count,
            .image_radius = image_radius,
            .figures_count = figures_count,
            .data = data_set,
            .labels = data_set_labels,
    };
}

auto nn_figures_train(i32 frames_foreach_figure_count) {
    // --- Transform dataset --- //
    auto data_set = generate_square_circle_triangle_images(frames_foreach_figure_count);

    std::vector<f32> train_data_in(data_set.data.size());
    std::vector<f32> train_data_out(data_set.labels.size());

    size_t images_count = data_set.images_count;
    size_t inputs_count = data_set.image_radius * data_set.image_radius;
    size_t outputs_count = data_set.figures_count;

    auto floatifizer = [](u8 x) { return (f32) x; };

    std::transform(data_set.data.begin(), data_set.data.end(), train_data_in.begin(), floatifizer);
    std::transform(data_set.labels.begin(), data_set.labels.end(), train_data_out.begin(), floatifizer);

    // --- Train neural network --- //
    NeuralNetwork nn({inputs_count, 16, 16, 16, outputs_count});

    StdoutTrainingObserver observer;
    auto training_result = nn.train({
                                            .train_data_count = images_count,
                                            .input_data = train_data_in.data(),
                                            .output_data = train_data_out.data(),
                                    }, &observer);

    print_training_info_to_stdout(training_result);

    std::cout << "[INFO] Test\n";
    for (int figure = 0; figure < 3; figure++) {
        std::vector<f32> in;

        for (int j = 0; j < 49; j++) {
            in.push_back(train_data_in[figure * 49 * (images_count + 5) + j]);
        }

        auto res = nn.predict(in);

        std::cout << std::format("{} is (o: {:.3f}, #: {:>.3f} ^: {:>.3f}) \n",
                                 "o#^"[figure], res.at(0), res.at(1), res.at(2));
    }

    auto out_filename = std::format("../figures_model_o#^_x{}_each.bin", frames_foreach_figure_count);
    save_binary(out_filename.c_str(), NeuralNetworkModel::serialize(training_result.model));

    std::cout << std::format("[INFO] Model saved as \"{}\"\n", out_filename);

    std::cout << "\n";
}

auto nn_figures_test(i32 frames_foreach_figure_count) {
    auto out_filename = std::format("../figures_model_o#^_x{}_each.bin", frames_foreach_figure_count);

    auto model_data = load_binary(out_filename.c_str());
    auto model = NeuralNetworkModel::deserialize(model_data);

    NeuralNetwork nn(model);

    auto predict_and_print = [](NeuralNetwork &n, const char *lbl, std::vector<f32> d) {
        auto res = n.predict(d);
        std::cout << std::format("expected -> {} | predicted -> (o: {:.3f}, #: {:>.3f} ^: {:>.3f}) \n",
                                 lbl, res.at(0), res.at(1), res.at(2));
    };

#define X 1
    predict_and_print(nn, "#", {
            0, 0, 0, 0, 0, 0, 0,
            0, X, X, X, X, X, 0,
            0, X, 0, 0, 0, X, 0,
            0, X, 0, 0, 0, X, 0,
            0, X, 0, 0, 0, X, 0,
            0, X, X, X, X, X, 0,
            0, 0, 0, 0, 0, 0, 0,
    });

    predict_and_print(nn, "o", {
            0, 0, 0, 0, 0, 0, 0,
            0, 0, X, X, X, 0, 0,
            0, X, 0, 0, 0, X, 0,
            0, X, 0, X, 0, X, 0,
            0, X, 0, 0, 0, X, 0,
            0, 0, X, X, X, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
    });

    predict_and_print(nn, "^", {
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, X, 0, 0, 0,
            0, 0, X, 0, X, 0, 0,
            0, X, 0, 0, 0, X, 0,
            X, 0, 0, 0, 0, 0, X,
            X, X, X, X, X, X, X,
            0, 0, 0, 0, 0, 0, 0,
    });
#undef X
}

auto main() -> int {
    nn_figures_train(5);
    nn_figures_test(5);
//
//    nn_xor_train_and_save();
//    nn_xor_test();

    return 0;
}

