#include "exactsearch/batch.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using exactsearch::BatchExecutor;
using exactsearch::BatchMethod;
using exactsearch::KDTree;
using exactsearch::Matrix;
using exactsearch::Neighbor;
using exactsearch::SearchResult;

std::size_t batch_assertions = 0;
std::size_t batch_groups = 0;

void check(bool condition, const char* expression, int line) {
    ++batch_assertions;
    if (!condition) {
        std::ostringstream message;
        message << "batch test line " << line << ": " << expression;
        throw std::runtime_error(message.str());
    }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

template <class Function>
void rejects(Function&& function, int line) {
    ++batch_assertions;
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    std::ostringstream message;
    message << "batch test line " << line << ": expected a standard exception";
    throw std::runtime_error(message.str());
}

#define REJECTS(expression) rejects([&] { static_cast<void>(expression); }, __LINE__)

// Independent full-sort reference. All fixtures have small integer coordinates,
// so every subtraction, product and sum is exactly representable in double.
std::vector<Neighbor> oracle(const Matrix& data, const std::vector<float>& query,
                             std::size_t k) {
    std::vector<Neighbor> candidates;
    for (std::size_t row = 0; row < data.rows(); ++row) {
        double sum = 0;
        for (std::size_t j = 0; j < data.dim(); ++j) {
            const double delta = static_cast<double>(data.values()[row * data.dim() + j]) -
                                 static_cast<double>(query[j]);
            sum += delta * delta;
        }
        candidates.push_back({static_cast<std::uint32_t>(row), sum});
    }
    std::sort(candidates.begin(), candidates.end(), [](const Neighbor& a, const Neighbor& b) {
        return a.distance2 < b.distance2 || (a.distance2 == b.distance2 && a.id < b.id);
    });
    if (candidates.size() > k) candidates.resize(k);
    return candidates;
}

void check_neighbors(const std::vector<Neighbor>& observed,
                     const std::vector<Neighbor>& expected) {
    CHECK(observed.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        CHECK(observed[i].id == expected[i].id);
        CHECK(observed[i].distance2 == expected[i].distance2);
        CHECK(std::isfinite(observed[i].distance2));
    }
}

void check_results(const std::vector<SearchResult>& results, const Matrix& data,
                   const KDTree* tree, const Matrix& queries, std::size_t k,
                   BatchMethod method) {
    CHECK(results.size() == queries.rows());
    for (std::size_t i = 0; i < queries.rows(); ++i) {
        const std::vector<float> query(queries.row(i), queries.row(i) + queries.dim());
        const auto expected = oracle(data, query, k);
        check_neighbors(results[i].neighbors, expected);
        // Results use an independent oracle. Counters use the corresponding
        // serial path, which must execute identically for each separate query.
        const auto serial = method == BatchMethod::Flat ?
            exactsearch::search_flat(data, query, k) : tree->search(query, k);
        const auto& stats = results[i].stats;
        CHECK(stats.distance_evaluations == serial.stats.distance_evaluations);
        CHECK(stats.nodes_visited == serial.stats.nodes_visited);
        CHECK(stats.box_evaluations == serial.stats.box_evaluations);
        CHECK(stats.heap_updates == serial.stats.heap_updates);
        CHECK(stats.distance_evaluations <= data.rows());
        CHECK(stats.heap_updates <= stats.distance_evaluations);
        if (k == 0 || data.rows() == 0) {
            CHECK(stats.distance_evaluations == 0);
            CHECK(stats.nodes_visited == 0);
            CHECK(stats.box_evaluations == 0);
            CHECK(stats.heap_updates == 0);
        }
    }
}

struct Fixture {
    std::shared_ptr<const Matrix> data;
    std::shared_ptr<const KDTree> tree;
    Matrix queries;
};

Fixture random_fixture() {
    std::mt19937 generator(0xBA7C1234U);
    constexpr std::size_t n = 73, dim = 7, q = 17;
    std::vector<float> values(n * dim);
    std::vector<float> queries(q * dim);
    for (auto& value : values) value = static_cast<int>(generator() % 19) - 9;
    for (auto& value : queries) value = static_cast<int>(generator() % 29) - 14;
    std::copy_n(values.begin(), dim, values.begin() + 8 * dim);
    std::copy_n(values.begin(), dim, queries.begin());
    auto data = std::make_shared<const Matrix>(n, dim, std::move(values));
    auto tree = std::make_shared<const KDTree>(data, 3);
    return {data, tree, Matrix(q, dim, std::move(queries))};
}

void contract_rejections_and_recovery() {
    auto fixture = random_fixture();
    REJECTS(BatchExecutor(nullptr, nullptr, 1));
    REJECTS(BatchExecutor(fixture.data, fixture.tree, 0));
    REJECTS(BatchExecutor(fixture.data, fixture.tree, 65));
    auto other_data = std::make_shared<const Matrix>(2, 7, std::vector<float>(14, 1));
    auto other_tree = std::make_shared<const KDTree>(other_data, 1);
    REJECTS(BatchExecutor(fixture.data, other_tree, 2));
    BatchExecutor flat_only(fixture.data, nullptr, 2);
    REJECTS(flat_only.search(fixture.queries, 1, BatchMethod::KDTree));
    check_results(flat_only.search(fixture.queries, 3, BatchMethod::Flat),
                  *fixture.data, nullptr, fixture.queries, 3, BatchMethod::Flat);

    BatchExecutor executor(fixture.data, fixture.tree, 4);
    Matrix wrong_dimension(2, 8, std::vector<float>(16, 0));
    Matrix wrong_empty_dimension(0, 8, {});
    for (const auto method : {BatchMethod::Flat, BatchMethod::KDTree}) {
        for (std::size_t k : {std::size_t{0}, std::size_t{1}}) {
            REJECTS(executor.search(wrong_dimension, k, method));
            REJECTS(executor.search(wrong_empty_dimension, k, method));
        }
        check_results(executor.search(fixture.queries, 10, method),
                      *fixture.data, fixture.tree.get(), fixture.queries, 10, method);
    }
}

void every_thread_count_and_query_local_stats() {
    auto fixture = random_fixture();
    for (std::size_t threads : {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
        BatchExecutor executor(fixture.data, fixture.tree, threads);
        CHECK(executor.threads() == threads);
        for (const auto method : {BatchMethod::Flat, BatchMethod::KDTree}) {
            for (std::size_t k : {std::size_t{0}, std::size_t{1}, std::size_t{10}, std::size_t{100}}) {
                const auto result = executor.search(fixture.queries, k, method);
                check_results(result, *fixture.data, fixture.tree.get(), fixture.queries, k, method);
            }
            // Interleave a different k and method; prior output and its counters
            // must remain owned by the earlier call and must not accumulate.
            const auto before = executor.search(fixture.queries, 3, method);
            static_cast<void>(executor.search(fixture.queries, 15,
                method == BatchMethod::Flat ? BatchMethod::KDTree : BatchMethod::Flat));
            check_results(before, *fixture.data, fixture.tree.get(), fixture.queries, 3, method);
            check_results(executor.search(fixture.queries, 3, method),
                          *fixture.data, fixture.tree.get(), fixture.queries, 3, method);
        }
    }
}

void empty_inputs_small_batches_and_duplicates() {
    auto duplicate_data = std::make_shared<const Matrix>(35, 2, std::vector<float>(70, 2));
    auto duplicate_tree = std::make_shared<const KDTree>(duplicate_data, 1);
    const Matrix two_queries(2, 2, {2, 2, -1, 4});
    const Matrix empty_queries(0, 2, {});
    BatchExecutor executor(duplicate_data, duplicate_tree, 8);
    for (const auto method : {BatchMethod::Flat, BatchMethod::KDTree}) {
        for (std::size_t k : {std::size_t{0}, std::size_t{1}, std::size_t{20}, std::size_t{100}}) {
            CHECK(executor.search(empty_queries, k, method).empty());
            const auto result = executor.search(two_queries, k, method);
            check_results(result, *duplicate_data, duplicate_tree.get(), two_queries, k, method);
            if (k > 0) CHECK(result[0].neighbors.front().id == 0);
        }
    }
    auto empty_data = std::make_shared<const Matrix>(0, 2, std::vector<float>{});
    auto empty_tree = std::make_shared<const KDTree>(empty_data, 1);
    BatchExecutor empty_executor(empty_data, empty_tree, 4);
    for (const auto method : {BatchMethod::Flat, BatchMethod::KDTree}) {
        check_results(empty_executor.search(two_queries, 10, method),
                      *empty_data, empty_tree.get(), two_queries, 10, method);
    }
}

void concurrent_callers_and_all_futures_joined() {
    auto fixture = random_fixture();
    const Matrix second_queries(3, 7, {0, 0, 0, 0, 0, 0, 0,
                                      1, 2, 3, 4, 5, 6, 7,
                                     -1, -2, -3, -4, -5, -6, -7});
    for (std::size_t threads : {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
        BatchExecutor executor(fixture.data, fixture.tree, threads);
        for (int round = 0; round < 4; ++round) {
            std::promise<void> gate;
            const auto start = gate.get_future().share();
            const auto first_method = round % 2 == 0 ? BatchMethod::Flat : BatchMethod::KDTree;
            const auto second_method = round % 2 == 0 ? BatchMethod::KDTree : BatchMethod::Flat;
            const std::size_t first_k = round == 0 ? 0 : 9;
            const std::size_t second_k = round == 1 ? 100 : 2;
            auto first = std::async(std::launch::async, [&] {
                start.wait();
                return executor.search(fixture.queries, first_k, first_method);
            });
            // If launching caller two fails, release caller one before its
            // future is destroyed; otherwise it could wait on an unopened gate.
            struct ReleaseGate {
                std::promise<void>& promise;
                bool released = false;
                ~ReleaseGate() {
                    if (!released) {
                        try { promise.set_value(); } catch (...) {}
                    }
                }
            } release{gate};
            auto second = std::async(std::launch::async, [&] {
                start.wait();
                return executor.search(second_queries, second_k, second_method);
            });
            gate.set_value();
            release.released = true;
            std::vector<SearchResult> first_result, second_result;
            std::exception_ptr failure;
            // Always get both futures, including when one call reports an error.
            // Executor, queries and captures remain alive throughout both calls.
            try { first_result = first.get(); }
            catch (...) { failure = std::current_exception(); }
            try { second_result = second.get(); }
            catch (...) { if (!failure) failure = std::current_exception(); }
            if (failure) std::rethrow_exception(failure);
            check_results(first_result, *fixture.data, fixture.tree.get(), fixture.queries,
                          first_k, first_method);
            check_results(second_result, *fixture.data, fixture.tree.get(), second_queries,
                          second_k, second_method);
        }
        // Both calling threads have finished before reuse and destruction.
        check_results(executor.search(fixture.queries, 4, BatchMethod::KDTree),
                      *fixture.data, fixture.tree.get(), fixture.queries, 4, BatchMethod::KDTree);
    }
}

void repeated_lifecycle_and_retained_owners() {
    auto fixture = random_fixture();
    for (std::size_t iteration = 0; iteration < 20; ++iteration) {
        const std::size_t threads = std::size_t{1} << (iteration % 4);
        BatchExecutor executor(fixture.data, fixture.tree, threads);
        for (std::size_t invocation = 0; invocation < 3; ++invocation) {
            const auto method = invocation % 2 == 0 ? BatchMethod::Flat : BatchMethod::KDTree;
            check_results(executor.search(fixture.queries, invocation + 1, method),
                          *fixture.data, fixture.tree.get(), fixture.queries, invocation + 1, method);
        }
        // Synchronous search has drained its tasks. Scope exit must join every
        // persistent worker; no forbidden concurrent destructor test is used.
    }

    std::weak_ptr<const Matrix> weak_data;
    std::weak_ptr<const KDTree> weak_tree;
    {
        auto data = std::make_shared<const Matrix>(3, 1, std::vector<float>{1, -1, 3});
        auto tree = std::make_shared<const KDTree>(data, 1);
        weak_data = data;
        weak_tree = tree;
        BatchExecutor executor(data, tree, 2);
        data.reset();
        tree.reset();
        CHECK(!weak_data.expired());
        CHECK(!weak_tree.expired());
        const Matrix queries(1, 1, {0});
        for (const auto method : {BatchMethod::Flat, BatchMethod::KDTree}) {
            const auto result = executor.search(queries, 3, method);
            CHECK(result.size() == 1);
            check_neighbors(result[0].neighbors, {{0, 1}, {1, 1}, {2, 9}});
        }
    }
    CHECK(weak_data.expired());
    CHECK(weak_tree.expired());
}

void group(const char* name, const std::function<void()>& function) {
    const auto before = batch_assertions;
    function();
    ++batch_groups;
    std::cout << "[PASS] " << name << " (" << batch_assertions - before << " assertions)\n";
}

} // namespace

void run_batch_tests() {
    group("batch invalid contracts and recovery", contract_rejections_and_recovery);
    group("batch 1/2/4/8 workers match independent oracle and serial counters", every_thread_count_and_query_local_stats);
    group("batch empty corpus/queries, fewer tasks than workers and duplicates", empty_inputs_small_batches_and_duplicates);
    group("batch concurrent callers with every future joined", concurrent_callers_and_all_futures_joined);
    group("batch repeated construct/reuse/destruct and retained data ownership", repeated_lifecycle_and_retained_owners);
    std::cout << "Batch: " << batch_groups << " groups, " << batch_assertions << " assertions passed.\n";
}
