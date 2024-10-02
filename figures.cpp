#include "src/neural_network.h"

#include <iostream>
#include <format>
#include <chrono>

#include "src/utils.h"

struct StdoutTrainingObserver : public ITrainingObserver {
    auto on_epoch(const TrainingObserverData &data) -> void override {
        std::cout << std::format("epoch: {}; err: {}; data_set_index: {};\n",
                                 data.current_epoch, data.current_error, data.data_set_index);
    }
};

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

            if (img != 0) {
                figures[figure][y[0]] ^= (1 << x[0]);
                figures[figure][y[1]] ^= (1 << x[1]);
                figures[figure][y[2]] ^= (1 << x[2]);
                figures[figure][y[3]] ^= (1 << x[3]);
            }

            for (int i = 0; i < image_radius; i++) {
                for (int j = 0; j < image_radius; j++) {
                    data_set.push_back((figures[figure][i] & (0x40 >> j)) != 0);
                }
            }

            if (img != 0) {
                figures[figure][y[0]] ^= (1 << x[0]);
                figures[figure][y[1]] ^= (1 << x[1]);
                figures[figure][y[2]] ^= (1 << x[2]);
                figures[figure][y[3]] ^= (1 << x[3]);
            }

            data_set_labels.push_back(figure == 0);
            data_set_labels.push_back(figure == 1);
            data_set_labels.push_back(figure == 2);
        }
    }

    for (int k = 0; k < data_set_count; k++) {
        for (int i = 0; i < image_radius; i++) {
            for (int j = 0; j < image_radius; j++) {
                std::cout << ((data_set[k * 49 + i * 7 + j] != 0) ? "█" : "░");
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
    NeuralNetwork nn({inputs_count, 4, 8, 4, outputs_count});

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
            in.push_back(train_data_in[figure * 49 * (frames_foreach_figure_count) + j]);
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

        auto start = std::chrono::high_resolution_clock::now();
        auto res = n.predict(d);
        auto duration = std::chrono::high_resolution_clock::now() - start;

        std::cout << std::format("predict time: {}\n", duration);

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

    return 0;
}

