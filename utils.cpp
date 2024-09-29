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
