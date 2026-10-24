#include "exactsearch/search.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Defined in test_io_batch.cpp. This executable verifies the core first.
void run_io_batch_tests();

namespace {

using exactsearch::BoundedTopK;
using exactsearch::KDTree;
using exactsearch::Matrix;
using exactsearch::Neighbor;
using exactsearch::SearchResult;

std::size_t assertions = 0;
std::size_t passed_groups = 0;

void check(bool condition, const char* expression, int line) {
    ++assertions;
    if (!condition) {
        std::ostringstream message;
        message << "line " << line << ": " << expression;
        throw std::runtime_error(message.str());
    }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

template <class Exception, class Function>
void expect_throw(Function&& function, int line) {
    ++assertions;
    try {
        function();
    } catch (const Exception&) {
        return;
    } catch (const std::exception& error) {
        std::ostringstream message;
        message << "line " << line << ": wrong exception type: " << error.what();
        throw std::runtime_error(message.str());
    }
    std::ostringstream message;
    message << "line " << line << ": expected exception was not thrown";
    throw std::runtime_error(message.str());
}

#define EXPECT_THROW(type, expression) \
    expect_throw<type>([&] { static_cast<void>(expression); }, __LINE__)

// This reference deliberately uses neither production better(), squared_l2(),
// BoundedTopK nor search_sort(). Integer/uint8 cases are exactly representable.
// Long double additionally detects float subtraction/squaring overflow in the
// numerical tests. Those cases permit final double accumulation rounding.
std::vector<Neighbor> oracle(const Matrix& data,
                             const std::vector<float>& query,
                             std::size_t k) {
    std::vector<Neighbor> result;
    result.reserve(data.rows());
    for (std::size_t row = 0; row < data.rows(); ++row) {
        long double sum = 0.0L;
        for (std::size_t column = 0; column < data.dim(); ++column) {
            const long double delta = static_cast<long double>(query[column]) -
                static_cast<long double>(data.values()[row * data.dim() + column]);
            sum += delta * delta;
        }
        result.push_back({static_cast<std::uint32_t>(row), static_cast<double>(sum)});
    }
    std::sort(result.begin(), result.end(), [](const Neighbor& a, const Neighbor& b) {
        if (a.distance2 != b.distance2) return a.distance2 < b.distance2;
        return a.id < b.id;
    });
    if (result.size() > k) result.resize(k);
    return result;
}

bool near(double observed, double expected) {
    if (observed == expected) return true;
    if (!std::isfinite(observed) || !std::isfinite(expected)) return false;
    // Absolute scaling by 1 would hide complete loss of subnormal distances.
    const double scale = std::max(std::abs(observed), std::abs(expected));
    return std::abs(observed - expected) <=
        32.0 * std::numeric_limits<double>::epsilon() * scale;
}

void check_neighbors(const std::vector<Neighbor>& actual,
                     const std::vector<Neighbor>& expected,
                     bool permit_rounding = false) {
    CHECK(actual.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        CHECK(actual[i].id == expected[i].id);
        CHECK(std::isfinite(actual[i].distance2));
        CHECK(actual[i].distance2 >= 0.0);
        CHECK(permit_rounding ? near(actual[i].distance2, expected[i].distance2)
                              : actual[i].distance2 == expected[i].distance2);
        if (i > 0) {
            CHECK(actual[i - 1].distance2 < actual[i].distance2 ||
                  (actual[i - 1].distance2 == actual[i].distance2 &&
                   actual[i - 1].id < actual[i].id));
        }
    }
}

void check_stats(const SearchResult& result, std::size_t n,
                 std::size_t k, bool exhaustive) {
    CHECK(result.stats.distance_evaluations <= n);
    CHECK(result.stats.heap_updates <= result.stats.distance_evaluations);
    if (k == 0 || n == 0) {
        CHECK(result.stats.distance_evaluations == 0);
    } else if (exhaustive) {
        CHECK(result.stats.distance_evaluations == n);
    }
}

void check_methods(const std::shared_ptr<const Matrix>& data,
                   const std::vector<float>& query, std::size_t k,
                   std::size_t leaf_size, bool permit_rounding = false) {
    const auto expected = oracle(*data, query, k);
    const auto sorted = exactsearch::search_sort(*data, query, k);
    const auto flat = exactsearch::search_flat(*data, query, k);
    KDTree tree(data, leaf_size);
    const auto indexed = tree.search(query, k);
    check_neighbors(sorted.neighbors, expected, permit_rounding);
    check_neighbors(flat.neighbors, expected, permit_rounding);
    check_neighbors(indexed.neighbors, expected, permit_rounding);
    check_stats(sorted, data->rows(), k, true);
    check_stats(flat, data->rows(), k, true);
    check_stats(indexed, data->rows(), k, false);
    CHECK(indexed.stats.nodes_visited <= tree.node_count());
    CHECK(tree.leaf_size() == leaf_size);
    CHECK(&tree.data() == data.get());
}

void manual_two_dimensions() {
    auto data = std::make_shared<const Matrix>(6, 2, std::vector<float>{
        0, 0, 3, 4, -3, 4, 0, 1, 0, -1, 0, 0});
    const std::vector<float> origin{0, 0};
    const std::vector<Neighbor> expected{{0, 0}, {5, 0}, {3, 1},
                                          {4, 1}, {1, 25}, {2, 25}};
    check_neighbors(oracle(*data, origin, 20), expected);
    check_neighbors(exactsearch::search_sort(*data, origin, 20).neighbors, expected);
    check_neighbors(exactsearch::search_flat(*data, origin, 20).neighbors, expected);
    for (std::size_t leaf : {std::size_t{1}, std::size_t{2}, std::size_t{16}}) {
        KDTree tree(data, leaf);
        check_neighbors(tree.search(origin, 20).neighbors, expected);
        const std::vector<Neighbor> shifted{{1, 0}, {3, 18}, {0, 25},
                                          {5, 25}, {4, 34}, {2, 36}};
        check_neighbors(tree.search({3, 4}, 6).neighbors, shifted);
    }
    CHECK(exactsearch::squared_l2(data->row(0), data->row(1), 2) == 25.0);
    CHECK(exactsearch::squared_l2(data->row(1), data->row(2), 2) == 36.0);
    CHECK(exactsearch::squared_l2(data->row(0), data->row(5), 2) == 0.0);
}

void heap_prefix_invariants() {
    BoundedTopK zero(0);
    CHECK(!zero.push({0, 1}));
    CHECK(zero.size() == 0);
    CHECK(zero.sorted().empty());
    CHECK(zero.invariant());
    EXPECT_THROW(std::out_of_range, zero.worst());

    std::vector<Neighbor> stream{{90, 9}, {80, 9}, {70, 4}, {60, 4},
                                 {50, 1}, {40, 1}, {30, 0}, {20, 0},
                                 {10, 9}, {1, 0}, {0, 0}};
    std::mt19937 generator(271828U);
    for (std::uint32_t i = 100; i < 300; ++i) {
        stream.push_back({i, static_cast<double>(generator() % 17)});
    }
    for (std::size_t capacity : {std::size_t{1}, std::size_t{2},
                                  std::size_t{7}, std::size_t{500}}) {
        BoundedTopK heap(capacity);
        CHECK(heap.invariant());
        EXPECT_THROW(std::out_of_range, heap.worst());
        std::vector<Neighbor> seen;
        for (const auto candidate : stream) {
            const auto before = heap.sorted();
            seen.push_back(candidate);
            auto expected = seen;
            std::sort(expected.begin(), expected.end(), [](const Neighbor& a,
                                                           const Neighbor& b) {
                return a.distance2 < b.distance2 ||
                    (a.distance2 == b.distance2 && a.id < b.id);
            });
            if (expected.size() > capacity) expected.resize(capacity);
            const bool changed = heap.push(candidate);
            CHECK(heap.invariant());
            CHECK(heap.size() == std::min(capacity, seen.size()));
            CHECK(heap.full() == (heap.size() == capacity));
            check_neighbors(heap.sorted(), expected);
            CHECK(heap.worst().id == expected.back().id);
            CHECK(heap.worst().distance2 == expected.back().distance2);
            const bool expected_changed = before.size() != expected.size() ||
                !std::equal(before.begin(), before.end(), expected.begin(),
                    [](const Neighbor& a, const Neighbor& b) {
                        return a.id == b.id && a.distance2 == b.distance2;
                    });
            CHECK(changed == expected_changed);
        }
    }
    BoundedTopK tied(2);
    CHECK(tied.push({9, 1}));
    CHECK(tied.push({8, 1}));
    CHECK(tied.push({3, 1}));
    CHECK(!tied.push({12, 1}));
    check_neighbors(tied.sorted(), {{3, 1}, {8, 1}});
    for (double bad : {std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::infinity(), -1.0}) {
        EXPECT_THROW(std::invalid_argument, tied.push({0, bad}));
        CHECK(tied.invariant());
        check_neighbors(tied.sorted(), {{3, 1}, {8, 1}});
    }
}

void matrix_and_query_validation() {
    EXPECT_THROW(std::invalid_argument, Matrix(1, 0, {}));
    EXPECT_THROW(std::invalid_argument, Matrix(0, 0, {}));
    EXPECT_THROW(std::invalid_argument, Matrix(2, 3, {1, 2, 3}));
    EXPECT_THROW(std::invalid_argument, Matrix(0, 2, {1}));
    EXPECT_THROW(std::length_error,
                 Matrix(std::numeric_limits<std::size_t>::max(), 2, {}));
    EXPECT_THROW(std::length_error,
                 Matrix(2, std::numeric_limits<std::size_t>::max(), {}));
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    for (float bad : {nan, infinity, -infinity}) {
        EXPECT_THROW(std::invalid_argument, Matrix(1, 2, {0, bad}));
    }
    auto data = std::make_shared<const Matrix>(2, 2,
                                              std::vector<float>{1, 2, 3, 4});
    CHECK(data->row(1) == data->values().data() + 2);
    CHECK(data->row(0)[1] == 2.0f);
    EXPECT_THROW(std::out_of_range, data->row(2));
    EXPECT_THROW(std::out_of_range, data->row(std::numeric_limits<std::size_t>::max()));
    EXPECT_THROW(std::invalid_argument, KDTree(nullptr, 1));
    EXPECT_THROW(std::invalid_argument, KDTree(data, 0));
    KDTree tree(data, 1);
    const std::vector<std::vector<float>> invalid_queries{
        {}, {1}, {1, 2, 3}, {nan, 0}, {0, infinity}, {-infinity, 0}};
    for (const auto& query : invalid_queries) {
        EXPECT_THROW(std::invalid_argument, exactsearch::validate_query(*data, query));
        for (std::size_t k : {std::size_t{0}, std::size_t{1}}) {
            EXPECT_THROW(std::invalid_argument, exactsearch::search_sort(*data, query, k));
            EXPECT_THROW(std::invalid_argument, exactsearch::search_flat(*data, query, k));
            EXPECT_THROW(std::invalid_argument, tree.search(query, k));
        }
    }
    exactsearch::validate_query(*data, {0, 0});
}

void empty_k_and_duplicate_boundaries() {
    auto empty = std::make_shared<const Matrix>(0, 3, std::vector<float>{});
    EXPECT_THROW(std::out_of_range, empty->row(0));
    for (std::size_t k : {std::size_t{0}, std::size_t{1}, std::size_t{100}}) {
        check_methods(empty, {0, 0, 0}, k, 1);
    }
    auto one = std::make_shared<const Matrix>(1, 1, std::vector<float>{4});
    for (std::size_t k : {std::size_t{0}, std::size_t{1}, std::size_t{2},
                          std::numeric_limits<std::size_t>::max()}) {
        check_methods(one, {-2}, k, 1);
    }
    auto duplicates = std::make_shared<const Matrix>(65, 3,
                                                    std::vector<float>(65 * 3, 3));
    for (std::size_t leaf : {std::size_t{1}, std::size_t{8}, std::size_t{100}}) {
        for (std::size_t k : {std::size_t{0}, std::size_t{1}, std::size_t{17},
                              std::size_t{65}, std::size_t{80}}) {
            check_methods(duplicates, {3, 3, 3}, k, leaf);
            check_methods(duplicates, {-1, 5, 3}, k, leaf);
        }
    }
    // Equal lower bounds must remain searchable: the second branch can contain
    // the smaller ID at exactly the same distance as the incumbent candidate.
    auto ties = std::make_shared<const Matrix>(2, 1, std::vector<float>{1, -1});
    check_methods(ties, {0}, 1, 1);
    check_neighbors(KDTree(ties, 1).search({0}, 1).neighbors, {{0, 1}});
}

void numerical_extremes() {
    const float maximum = std::numeric_limits<float>::max();
    const float tiny = std::numeric_limits<float>::denorm_min();
    auto data = std::make_shared<const Matrix>(4, 4, std::vector<float>{
        maximum, -maximum, tiny, -0.0f,
        -maximum, maximum, 0, 0,
        0, 0, tiny, 0,
        maximum / 2, -maximum / 2, 0, 0});
    const std::vector<float> query{maximum, -maximum, tiny, 0};
    const long double delta = static_cast<long double>(maximum) * 2.0L;
    const double expected = static_cast<double>(delta * delta * 2.0L +
                                               static_cast<long double>(tiny) * tiny);
    const double actual = exactsearch::squared_l2(data->row(0), data->row(1), 4);
    CHECK(std::isfinite(actual));
    CHECK(near(actual, expected));
    CHECK(exactsearch::squared_l2(data->row(0), data->row(0), 4) == 0.0);
    for (std::size_t leaf : {std::size_t{1}, std::size_t{2}, std::size_t{16}}) {
        check_methods(data, query, 4, leaf, true);
        check_methods(data, {0, 0, 0, 0}, 4, leaf, true);
    }
    auto subnormal = std::make_shared<const Matrix>(3, 1,
                                      std::vector<float>{tiny, 2 * tiny, 0});
    const double squared_tiny = static_cast<double>(tiny) * static_cast<double>(tiny);
    CHECK(squared_tiny > 0.0);
    CHECK(exactsearch::squared_l2(subnormal->row(0), subnormal->row(2), 1) == squared_tiny);
    check_methods(subnormal, {0}, 3, 1, true);
}

void one_dimensional_collinear_and_high_dimensional() {
    auto line = std::make_shared<const Matrix>(9, 1,
                        std::vector<float>{-4, -3, -2, -1, 0, 1, 2, 3, 4});
    for (const auto query : {-8.0f, -0.5f, 0.0f, 2.5f, 8.0f}) {
        for (std::size_t k : {std::size_t{1}, std::size_t{4}, std::size_t{12}}) {
            check_methods(line, {query}, k, 2);
        }
    }
    std::vector<float> collinear;
    for (int i = -32; i <= 32; ++i) {
        collinear.insert(collinear.end(), {static_cast<float>(i),
            static_cast<float>(2 * i), static_cast<float>(-i), 7.0f});
    }
    auto degenerate = std::make_shared<const Matrix>(65, 4, std::move(collinear));
    for (std::size_t k : {std::size_t{1}, std::size_t{20}, std::size_t{65}}) {
        check_methods(degenerate, {1, 2, -1, 7}, k, 3);
        check_methods(degenerate, {2, -3, 5, -10}, k, 1);
    }
    std::vector<float> high(17 * 257, 0);
    for (std::size_t row = 0; row < 17; ++row) high[row * 257 + row * 13] = 1;
    auto sparse_high = std::make_shared<const Matrix>(17, 257, std::move(high));
    check_methods(sparse_high, std::vector<float>(257, 0), 17, 2);
    std::vector<float> exact_query(257, 0);
    exact_query[7 * 13] = 1;
    check_methods(sparse_high, exact_query, 7, 1);
}

void fixed_seed_integer_differential() {
    std::mt19937 generator(0x5EED1234U);
    const std::vector<std::size_t> sizes{1, 2, 3, 7, 31, 64, 127};
    const std::vector<std::size_t> dimensions{1, 2, 3, 8, 32};
    for (std::size_t n : sizes) {
        for (std::size_t dim : dimensions) {
            std::vector<float> values(n * dim);
            for (auto& value : values) value = static_cast<int>(generator() % 33) - 16;
            // Duplicate one whole row so that duplicate handling is exercised
            // in random data, as well as in the all-equal adversarial fixture.
            if (n > 3) std::copy_n(values.begin(), dim, values.begin() + 3 * dim);
            auto data = std::make_shared<const Matrix>(n, dim, std::move(values));
            for (int query_index = 0; query_index < 4; ++query_index) {
                std::vector<float> query(dim);
                for (auto& value : query) value = static_cast<int>(generator() % 41) - 20;
                if (query_index == 0) std::copy_n(data->values().begin(), dim, query.begin());
                const std::vector<std::size_t> ks{0, 1, 2, n / 2, n, n + 7};
                for (std::size_t leaf : {std::size_t{1}, std::size_t{4}, std::size_t{16}}) {
                    for (std::size_t k : ks) check_methods(data, query, k, leaf);
                }
            }
        }
    }
}

void fixed_seed_uint8_descriptor_differential() {
    std::mt19937 generator(0xC001D00DU);
    for (std::size_t dim : {std::size_t{16}, std::size_t{64}, std::size_t{128}}) {
        constexpr std::size_t n = 193;
        std::vector<float> descriptors(n * dim);
        for (auto& value : descriptors) {
            value = static_cast<float>(static_cast<std::uint8_t>(generator() & 255U));
        }
        auto data = std::make_shared<const Matrix>(n, dim, std::move(descriptors));
        for (int repetition = 0; repetition < 8; ++repetition) {
            std::vector<float> query(dim);
            for (auto& value : query) {
                value = static_cast<float>(static_cast<std::uint8_t>(generator() & 255U));
            }
            // uint8 coordinate differences and their summed squares remain
            // exact in double; require bit-exact distances, not tolerance.
            for (std::size_t leaf : {std::size_t{1}, std::size_t{12}, std::size_t{256}}) {
                for (std::size_t k : {std::size_t{1}, std::size_t{10},
                                      std::size_t{193}, std::size_t{200}}) {
                    check_methods(data, query, k, leaf);
                }
            }
        }
    }
}

void data_lifetime_and_repeated_queries() {
    std::weak_ptr<const Matrix> lifetime;
    {
        std::unique_ptr<KDTree> tree;
        {
            auto data = std::make_shared<const Matrix>(3, 2,
                                            std::vector<float>{2, 0, 0, 0, -2, 0});
            lifetime = data;
            tree = std::make_unique<KDTree>(data, 1);
            data.reset();
        }
        CHECK(!lifetime.expired());
        CHECK(tree->data().rows() == 3);
        const auto first = tree->search({0, 0}, 3);
        check_neighbors(first.neighbors, {{1, 0}, {0, 4}, {2, 4}});
        check_neighbors(tree->search({-2, 0}, 1).neighbors, {{2, 0}});
        const auto repeated = tree->search({0, 0}, 3);
        check_neighbors(repeated.neighbors, first.neighbors);
        CHECK(repeated.stats.distance_evaluations == first.stats.distance_evaluations);
        CHECK(repeated.stats.nodes_visited == first.stats.nodes_visited);
        CHECK(repeated.stats.box_evaluations == first.stats.box_evaluations);
        CHECK(repeated.stats.heap_updates == first.stats.heap_updates);
    }
    CHECK(lifetime.expired());
}

void run_group(const char* name, const std::function<void()>& body) {
    const auto before = assertions;
    body();
    ++passed_groups;
    std::cout << "[PASS] " << name << " (" << assertions - before << " assertions)\n";
}

} // namespace

void run_batch_tests();

int main() {
    try {
        run_group("hand-verifiable 2D distances and ordering", manual_two_dimensions);
        run_group("bounded heap every-prefix invariant and tie replacement", heap_prefix_invariants);
        run_group("matrix, query and construction rejection", matrix_and_query_validation);
        run_group("empty input, extreme k, duplicates and equal-bound pruning", empty_k_and_duplicate_boundaries);
        run_group("finite extreme floats, subnormal and double accumulation", numerical_extremes);
        run_group("1D, collinear and sparse high-dimensional fixtures", one_dimensional_collinear_and_high_dimensional);
        run_group("fixed-seed independent integer differential", fixed_seed_integer_differential);
        run_group("fixed-seed uint8 descriptor exact differential", fixed_seed_uint8_descriptor_differential);
        run_group("index owns immutable data and query stats reset", data_lifetime_and_repeated_queries);
        std::cout << "Core: " << passed_groups << " groups, " << assertions << " assertions passed.\n";
        run_io_batch_tests();
        run_batch_tests();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] after " << passed_groups << " groups and " << assertions
                  << " assertions: " << error.what() << '\n';
    } catch (...) {
        std::cerr << "[FAIL] non-standard exception\n";
    }
    return 1;
}
