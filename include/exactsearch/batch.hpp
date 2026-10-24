#pragma once
#include "exactsearch/search.hpp"
#include <memory>
#include <vector>

namespace exactsearch {
enum class BatchMethod { Flat, KDTree };
// Persistent workers; each call owns its output, query heap and statistics.
// Calls may run concurrently; destroying the executor concurrently is forbidden.
class BatchExecutor {
public:
    BatchExecutor(std::shared_ptr<const Matrix> data,
                  std::shared_ptr<const KDTree> tree, std::size_t threads);
    ~BatchExecutor();
    BatchExecutor(const BatchExecutor&) = delete;
    BatchExecutor& operator=(const BatchExecutor&) = delete;
    std::vector<SearchResult> search(const Matrix& queries, std::size_t k,
                                     BatchMethod method);
    std::size_t threads() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace exactsearch
