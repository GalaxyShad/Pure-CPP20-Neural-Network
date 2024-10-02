#include "src/neural_network.h"

#include <iostream>
#include <format>
#include <chrono>

#include "src/utils.h"

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

auto main() -> int {
    nn_xor_train_and_save();
    nn_xor_test();

    return 0;
}

