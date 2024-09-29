#ifndef UTILS_H
#define UTILS_H

#include <vector>

#include "shorttypes.h"

auto save_binary(const char *filename, const std::vector<u8> &binary) -> void;

auto load_binary(const char *filename) -> std::vector<u8>;

#endif //UTILS_H
