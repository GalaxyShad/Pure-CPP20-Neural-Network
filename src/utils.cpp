#include "utils.h"

#include <fstream>

auto save_binary(const char *filename, const std::vector<u8> &binary) -> void {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char *>(binary.data()), binary.size());
}

auto load_binary(const char *filename) -> std::vector<u8> {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<u8> binary(size);
    file.read(reinterpret_cast<char *>(binary.data()), size);
    return binary;
}

auto print_training_info_to_stdout(const NeuralNetwork::TrainingResult &training_res) -> void {
    std::cout << "[TRAINING INFO]\n";
    std::cout << std::format("epochs:\t{:>10}\n", training_res.epoch_count);
    std::cout << std::format("error:\t{:>10}\n", training_res.error);
    std::cout << std::format("time millis:\t{:>10}\n", training_res.time);
}
