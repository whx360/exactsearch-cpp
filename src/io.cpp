#include "exactsearch/io.hpp"
#include <cstring>
#include <fstream>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace exactsearch {
namespace {
std::uint32_t decode32(const unsigned char* b) {
    return std::uint32_t(b[0]) | (std::uint32_t(b[1]) << 8) |
           (std::uint32_t(b[2]) << 16) | (std::uint32_t(b[3]) << 24);
}
void encode32(std::ostream& out, std::uint32_t x) {
    const unsigned char b[4] = {static_cast<unsigned char>(x),
        static_cast<unsigned char>(x >> 8), static_cast<unsigned char>(x >> 16),
        static_cast<unsigned char>(x >> 24)};
    out.write(reinterpret_cast<const char*>(b), 4);
}
}

// Input: file path and allocation cap. Output: validated Matrix.
// Time O(Nd), extra O(Nd), no unchecked dimension-based allocation.
Matrix load_fvecs(const std::string& path, std::size_t max_bytes) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "fvecs requires IEEE-754 binary32");
    std::ifstream in(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open fvecs: " + path);
    const auto end = in.tellg();
    if (end < 4 || static_cast<std::uint64_t>(end) > max_bytes)
        throw std::runtime_error("fvecs is empty, too small, or exceeds allocation cap");
    const auto bytes = static_cast<std::size_t>(end);
    in.seekg(0);
    unsigned char b[4];
    if (!in.read(reinterpret_cast<char*>(b), 4)) throw std::runtime_error("missing dimension");
    const std::uint32_t dim = decode32(b);
    if (dim == 0 || dim > 65536) throw std::runtime_error("invalid fvecs dimension");
    const std::size_t stride = 4ULL * (std::size_t(dim) + 1);
    if (bytes % stride != 0) throw std::runtime_error("truncated fvecs row");
    const std::size_t rows = bytes / stride;
    std::vector<float> values;
    values.reserve(rows * dim);
    in.seekg(0);
    for (std::size_t i = 0; i < rows; ++i) {
        if (!in.read(reinterpret_cast<char*>(b), 4) || decode32(b) != dim)
            throw std::runtime_error("inconsistent fvecs row dimension");
        for (std::size_t j = 0; j < dim; ++j) {
            if (!in.read(reinterpret_cast<char*>(b), 4)) throw std::runtime_error("truncated value");
            const auto bits = decode32(b);
            float value;
            std::memcpy(&value, &bits, 4);
            values.push_back(value);
        }
    }
    return Matrix(rows, dim, std::move(values));
}

// Input: validated matrix. Output: fvecs file. Time O(Nd), extra O(1).
void save_fvecs(const Matrix& data, const std::string& path) {
    if (data.dim() > 65536) throw std::invalid_argument("dimension exceeds fvecs limit");
    if (data.rows() == 0) throw std::invalid_argument("cannot encode empty fvecs");
    std::ofstream out(std::filesystem::u8path(path), std::ios::binary);
    if (!out) throw std::runtime_error("cannot create fvecs: " + path);
    for (std::size_t i = 0; i < data.rows(); ++i) {
        encode32(out, static_cast<std::uint32_t>(data.dim()));
        for (std::size_t j = 0; j < data.dim(); ++j) {
            std::uint32_t bits;
            const float x = data.row(i)[j];
            std::memcpy(&bits, &x, 4);
            encode32(out, bits);
        }
    }
    out.flush();
    if (!out) throw std::runtime_error("failed to write fvecs");
}
} // namespace exactsearch
