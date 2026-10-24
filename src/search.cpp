#include "exactsearch/search.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace exactsearch {

// Input: a row-major shape and its values. Output: validated immutable storage.
// Time O(rows * dim), additional space O(1) after taking ownership of values.
Matrix::Matrix(std::size_t rows, std::size_t dim, std::vector<float> values)
    : rows_(rows), dim_(dim), values_(std::move(values)) {
    if (dim_ == 0) {
        throw std::invalid_argument("Matrix dimension must be positive");
    }
    if (rows_ > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("Matrix row count exceeds uint32_t IDs");
    }
    if (rows_ > std::numeric_limits<std::size_t>::max() / dim_) {
        throw std::length_error("Matrix shape multiplication overflows size_t");
    }
    if (values_.size() != rows_ * dim_) {
        throw std::invalid_argument("Matrix shape does not match value count");
    }
    for (float value : values_) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Matrix values must be finite");
        }
    }
}

// Input: valid row ID. Output: pointer to dim() contiguous values. O(1) time/space.
const float* Matrix::row(std::size_t i) const {
    if (i >= rows_) {
        throw std::out_of_range("Matrix row index out of range");
    }
    return values_.data() + i * dim_;
}

// Lexicographic ascending order makes equal-distance results deterministic. O(1).
bool better(const Neighbor& a, const Neighbor& b) noexcept {
    return a.distance2 < b.distance2 ||
           (a.distance2 == b.distance2 && a.id < b.id);
}

// Input: two valid finite arrays of dim values. Output: squared L2 distance.
// Convert before subtraction; sum in dimension order. O(dim) time, O(1) space.
double squared_l2(const float* a, const float* b, std::size_t dim) noexcept {
    double result = 0.0;
    for (std::size_t j = 0; j < dim; ++j) {
        const double delta = static_cast<double>(a[j]) - static_cast<double>(b[j]);
        result += delta * delta;
    }
    return result;
}

// Input: data and a query. Output: none, or a descriptive validation exception.
// O(dim) time, O(1) space; runs even for empty data or k == 0.
void validate_query(const Matrix& data, const std::vector<float>& query) {
    if (query.size() != data.dim()) {
        throw std::invalid_argument("Query dimension does not match Matrix");
    }
    for (float value : query) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Query values must be finite");
        }
    }
}

// Storage grows with accepted candidates, not an untrusted requested capacity.
BoundedTopK::BoundedTopK(std::size_t capacity) : capacity_(capacity) {}

// Input: one finite, nonnegative-distance candidate. Output: whether heap changed.
// Root is the lexicographically worst retained item. Amortized O(log k) time,
// O(k) storage; a vector capacity growth can cost O(k) for an individual push.
bool BoundedTopK::push(Neighbor candidate) {
    if (capacity_ == 0) {
        return false;
    }
    if (!std::isfinite(candidate.distance2) || candidate.distance2 < 0.0) {
        throw std::invalid_argument("Candidate squared distance must be finite and nonnegative");
    }
    if (heap_.size() < capacity_) {
        heap_.push_back(candidate);
        std::size_t child = heap_.size() - 1;
        while (child > 0) {
            const std::size_t parent = (child - 1) / 2;
            if (!better(heap_[parent], heap_[child])) {
                break;
            }
            std::swap(heap_[parent], heap_[child]);
            child = parent;
        }
        return true;
    }
    if (!better(candidate, heap_.front())) {
        return false;
    }
    heap_.front() = candidate;
    std::size_t parent = 0;
    // A parent has a child exactly when parent < size / 2. This avoids 2*i overflow.
    while (parent < heap_.size() / 2) {
        const std::size_t left = parent * 2 + 1;
        const std::size_t right = left + 1;
        std::size_t worse_child = left;
        if (right < heap_.size() && better(heap_[left], heap_[right])) {
            worse_child = right;
        }
        if (!better(heap_[parent], heap_[worse_child])) {
            break;
        }
        std::swap(heap_[parent], heap_[worse_child]);
        parent = worse_child;
    }
    return true;
}

bool BoundedTopK::full() const noexcept {
    return heap_.size() == capacity_;
}

// Output: worst retained item. Empty heaps have no worst item. O(1).
const Neighbor& BoundedTopK::worst() const {
    if (heap_.empty()) {
        throw std::out_of_range("Empty Top-k heap has no worst neighbor");
    }
    return heap_.front();
}

// Output: ascending result copy, leaving the heap intact. O(k log k), O(k) space.
std::vector<Neighbor> BoundedTopK::sorted() const {
    std::vector<Neighbor> result = heap_;
    std::sort(result.begin(), result.end(), better);
    return result;
}

// Verify every parent is no better than either child. O(k) time, O(1) space.
bool BoundedTopK::invariant() const noexcept {
    if (heap_.size() > capacity_) {
        return false;
    }
    for (std::size_t child = 1; child < heap_.size(); ++child) {
        if (better(heap_[(child - 1) / 2], heap_[child])) {
            return false;
        }
    }
    return true;
}

// Reference: compute all N distances and fully sort. O(Nd + N log N), O(N) space.
SearchResult search_sort(const Matrix& data, const std::vector<float>& query,
                         std::size_t k) {
    validate_query(data, query);
    SearchResult result;
    const std::size_t count = std::min(k, data.rows());
    if (count == 0) {
        return result;
    }
    result.neighbors.reserve(data.rows());
    for (std::size_t i = 0; i < data.rows(); ++i) {
        result.neighbors.push_back({static_cast<std::uint32_t>(i),
                                    squared_l2(data.row(i), query.data(), data.dim())});
        ++result.stats.distance_evaluations;
    }
    std::sort(result.neighbors.begin(), result.neighbors.end(), better);
    result.neighbors.resize(count);
    return result;
}

// Exact scan: O(Nd + N log k + k log k) time, O(k) candidate space, k=min(k,N).
SearchResult search_flat(const Matrix& data, const std::vector<float>& query,
                         std::size_t k) {
    validate_query(data, query);
    SearchResult result;
    const std::size_t count = std::min(k, data.rows());
    if (count == 0) {
        return result;
    }
    BoundedTopK top(count);
    for (std::size_t i = 0; i < data.rows(); ++i) {
        const Neighbor candidate{static_cast<std::uint32_t>(i),
                                 squared_l2(data.row(i), query.data(), data.dim())};
        ++result.stats.distance_evaluations;
        if (top.push(candidate)) {
            ++result.stats.heap_updates;
        }
    }
    result.neighbors = top.sorted();
    return result;
}

// Input: shared immutable data and positive leaf size. Output: a balanced index.
// Expected O(Nd log N) build work; O(N + Md) index space for M tree nodes.
KDTree::KDTree(std::shared_ptr<const Matrix> data, std::size_t leaf_size)
    : data_(std::move(data)), leaf_size_(leaf_size) {
    if (!data_) {
        throw std::invalid_argument("KDTree data must not be null");
    }
    if (leaf_size_ == 0) {
        throw std::invalid_argument("KDTree leaf size must be positive");
    }
    ids_.resize(data_->rows());
    std::iota(ids_.begin(), ids_.end(), std::uint32_t{0});
    if (!ids_.empty()) {
        build(0, ids_.size());
    }
}

// Input: nonempty half-open ID range. Output: node number; splits by longest axis.
// Scan O(nd) and expected O(n) partition per node; balanced recursion depth O(log N).
int KDTree::build(std::size_t begin, std::size_t end) {
    if (nodes_.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("KDTree node count exceeds int node IDs");
    }
    const std::size_t next_count = nodes_.size() + 1;
    const std::size_t dim = data_->dim();
    if (next_count > std::numeric_limits<std::size_t>::max() / dim ||
        next_count * dim > lower_.max_size() ||
        next_count * dim > upper_.max_size()) {
        throw std::length_error("KDTree bounding box storage size overflows");
    }
    const int node = static_cast<int>(nodes_.size());
    nodes_.push_back(Node{begin, end, -1, -1});
    lower_.resize(next_count * dim);
    upper_.resize(next_count * dim);
    const std::size_t offset = static_cast<std::size_t>(node) * dim;
    const float* first = data_->row(ids_[begin]);
    for (std::size_t j = 0; j < dim; ++j) {
        lower_[offset + j] = first[j];
        upper_[offset + j] = first[j];
    }
    for (std::size_t i = begin + 1; i < end; ++i) {
        const float* point = data_->row(ids_[i]);
        for (std::size_t j = 0; j < dim; ++j) {
            lower_[offset + j] = std::min(lower_[offset + j], point[j]);
            upper_[offset + j] = std::max(upper_[offset + j], point[j]);
        }
    }
    if (end - begin <= leaf_size_) {
        return node;
    }
    std::size_t axis = 0;
    double widest = -1.0;
    for (std::size_t j = 0; j < dim; ++j) {
        const double width = static_cast<double>(upper_[offset + j]) -
                             static_cast<double>(lower_[offset + j]);
        if (width > widest) {
            widest = width;
            axis = j;
        }
    }
    const std::size_t middle = begin + (end - begin) / 2;
    std::nth_element(ids_.begin() + static_cast<std::ptrdiff_t>(begin),
                     ids_.begin() + static_cast<std::ptrdiff_t>(middle),
                     ids_.begin() + static_cast<std::ptrdiff_t>(end),
                     [this, axis](std::uint32_t a, std::uint32_t b) {
                         const float av = data_->row(a)[axis];
                         const float bv = data_->row(b)[axis];
                         return av < bv || (av == bv && a < b);
                     });
    // Keep only indices across recursion: vector growth can invalidate references.
    const int left = build(begin, middle);
    const int right = build(middle, end);
    nodes_[static_cast<std::size_t>(node)].left = left;
    nodes_[static_cast<std::size_t>(node)].right = right;
    return node;
}

// Input: valid node/query. Output: squared distance to its closed AABB, a lower
// bound for all enclosed points. O(dim) time, O(1) space, same sum order as L2.
double KDTree::box_distance(int node, const std::vector<float>& query) const noexcept {
    const std::size_t dim = data_->dim();
    const std::size_t offset = static_cast<std::size_t>(node) * dim;
    double result = 0.0;
    for (std::size_t j = 0; j < dim; ++j) {
        const double q = static_cast<double>(query[j]);
        double delta = 0.0;
        if (q < lower_[offset + j]) {
            delta = static_cast<double>(lower_[offset + j]) - q;
        } else if (q > upper_[offset + j]) {
            delta = q - static_cast<double>(upper_[offset + j]);
        }
        result += delta * delta;
    }
    return result;
}

// Exact branch-and-bound. nodes_visited counts considered nodes, including pruned
// ones; boxes count computed bounds; heap_updates counts accepted candidates.
// Worst O(Md + Nd + N log k), recursion O(log N); equal bounds must not be pruned.
void KDTree::visit(int node, double bound, const std::vector<float>& query,
                   BoundedTopK& top, SearchStats& stats) const {
    ++stats.nodes_visited;
    if (top.full() && bound > top.worst().distance2) {
        return;
    }
    const Node& current = nodes_[static_cast<std::size_t>(node)];
    if (current.left < 0) {
        for (std::size_t i = current.begin; i < current.end; ++i) {
            const std::uint32_t id = ids_[i];
            const Neighbor candidate{id, squared_l2(data_->row(id), query.data(), data_->dim())};
            ++stats.distance_evaluations;
            if (top.push(candidate)) {
                ++stats.heap_updates;
            }
        }
        return;
    }
    const double left_bound = box_distance(current.left, query);
    const double right_bound = box_distance(current.right, query);
    stats.box_evaluations += 2;
    if (left_bound <= right_bound) {
        visit(current.left, left_bound, query, top, stats);
        visit(current.right, right_bound, query, top, stats);
    } else {
        visit(current.right, right_bound, query, top, stats);
        visit(current.left, left_bound, query, top, stats);
    }
}

// Output: sorted exact top-k and query-local statistics. Index is never mutated.
// Worst traversal O(Md + Nd + N log k), plus O(k log k) output sort; O(k+log N) space.
SearchResult KDTree::search(const std::vector<float>& query, std::size_t k) const {
    validate_query(*data_, query);
    SearchResult result;
    const std::size_t count = std::min(k, data_->rows());
    if (count == 0) {
        return result;
    }
    BoundedTopK top(count);
    const double root_bound = box_distance(0, query);
    ++result.stats.box_evaluations;
    visit(0, root_bound, query, top, result.stats);
    result.neighbors = top.sorted();
    return result;
}

// Allocated container capacity bytes, excluding shared Matrix and allocator/object
// overhead. O(1); saturation prevents the estimate itself wrapping size_t.
std::size_t KDTree::index_bytes() const noexcept {
    std::size_t result = 0;
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    const auto add = [&result, maximum](std::size_t count, std::size_t width) {
        if (count > maximum / width || count * width > maximum - result) {
            result = maximum;
        } else {
            result += count * width;
        }
    };
    add(ids_.capacity(), sizeof(std::uint32_t));
    add(nodes_.capacity(), sizeof(Node));
    add(lower_.capacity(), sizeof(float));
    add(upper_.capacity(), sizeof(float));
    return result;
}

} // namespace exactsearch
