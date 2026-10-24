#pragma once
#include "exactsearch/search.hpp"
#include <string>

namespace exactsearch {
// Strict little-endian fvecs loader. Rejects truncation and inconsistent rows.
Matrix load_fvecs(const std::string& path, std::size_t max_bytes = 512ULL * 1024 * 1024);
void save_fvecs(const Matrix& data, const std::string& path);
} // namespace exactsearch
