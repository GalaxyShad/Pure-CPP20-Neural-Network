#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <iostream>
#include <format>

#include "shorttypes.h"
#include "neural_network.h"

auto save_binary(const char *filename, const std::vector<u8> &binary) -> void;

auto load_binary(const char *filename) -> std::vector<u8>;

auto print_training_info_to_stdout(const NeuralNetwork::TrainingResult& training_res) -> void;

#endif //UTILS_H
