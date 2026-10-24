#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace exactsearch {

// Immutable, row-major, finite float vectors. Empty rows are allowed; dim > 0.
class Matrix {
public:
    Matrix(std::size_t rows, std::size_t dim, std::vector<float> values);
    std::size_t rows() const noexcept { return rows_; }
    std::size_t dim() const noexcept { return dim_; }
    const float* row(std::size_t i) const;
    const std::vector<float>& values() const noexcept { return values_; }
private:
    std::size_t rows_, dim_;
    std::vector<float> values_;
};

struct Neighbor { std::uint32_t id; double distance2; };
bool better(const Neighbor& a, const Neighbor& b) noexcept;
double squared_l2(const float* a, const float* b, std::size_t dim) noexcept;
void validate_query(const Matrix& data, const std::vector<float>& query);

struct SearchStats {
    std::uint64_t distance_evaluations = 0;
    std::uint64_t nodes_visited = 0;
    std::uint64_t box_evaluations = 0;
    std::uint64_t heap_updates = 0;
};
struct SearchResult { std::vector<Neighbor> neighbors; SearchStats stats; };

// The root is the worst retained (distance2, id). Capacity zero discards input.
class BoundedTopK {
public:
    explicit BoundedTopK(std::size_t capacity);
    bool push(Neighbor candidate);
    bool full() const noexcept;
    std::size_t size() const noexcept { return heap_.size(); }
    const Neighbor& worst() const;
    std::vector<Neighbor> sorted() const;
    bool invariant() const noexcept;
private:
    std::size_t capacity_;
    std::vector<Neighbor> heap_;
};

SearchResult search_sort(const Matrix&, const std::vector<float>& query, std::size_t k);
SearchResult search_flat(const Matrix&, const std::vector<float>& query, std::size_t k);

class KDTree {
public:
    explicit KDTree(std::shared_ptr<const Matrix> data, std::size_t leaf_size = 16);
    SearchResult search(const std::vector<float>& query, std::size_t k) const;
    const Matrix& data() const noexcept { return *data_; }
    std::size_t node_count() const noexcept { return nodes_.size(); }
    std::size_t index_bytes() const noexcept;
    std::size_t leaf_size() const noexcept { return leaf_size_; }
private:
    struct Node {
        std::size_t begin = 0, end = 0;
        int left = -1, right = -1;
    };
    std::shared_ptr<const Matrix> data_;
    std::size_t leaf_size_;
    std::vector<std::uint32_t> ids_;
    std::vector<Node> nodes_;
    std::vector<float> lower_, upper_;
    int build(std::size_t begin, std::size_t end);
    double box_distance(int node, const std::vector<float>& query) const noexcept;
    void visit(int node, double bound, const std::vector<float>& query,
               BoundedTopK& top, SearchStats& stats) const;
};

} // namespace exactsearch
