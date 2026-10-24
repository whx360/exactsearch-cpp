#include "exactsearch/batch.hpp"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <exception>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace exactsearch {

// This pool serves only independent query ranges, rather than arbitrary public jobs.
struct BatchExecutor::Impl {
    std::shared_ptr<const Matrix> data;
    std::shared_ptr<const KDTree> tree;
    std::size_t thread_count;
    std::mutex mutex;
    std::condition_variable available;
    std::deque<std::packaged_task<void()>> pending;
    std::vector<std::thread> workers;
    bool stopping = false;

    Impl(std::shared_ptr<const Matrix> input, std::shared_ptr<const KDTree> index,
         std::size_t threads)
        : data(std::move(input)), tree(std::move(index)), thread_count(threads) {
        if (!data) {
            throw std::invalid_argument("BatchExecutor data must not be null");
        }
        if (tree && &tree->data() != data.get()) {
            throw std::invalid_argument("BatchExecutor tree must index the same Matrix object");
        }
        if (thread_count == 0 || thread_count > 64) {
            throw std::invalid_argument("BatchExecutor threads must be in [1,64]");
        }
        if (thread_count == 1) {
            return;
        }
        workers.reserve(thread_count);
        try {
            for (std::size_t i = 0; i < thread_count; ++i) {
                workers.emplace_back([this] { worker_loop(); });
            }
        } catch (...) {
            // A partially constructed Impl has no destructor call. Join here before
            // its mutex, condition variable, queue and thread vector are destroyed.
            stop_and_join();
            throw;
        }
    }

    ~Impl() {
        stop_and_join();
    }

    // Drain already queued work before stopping; never hold mutex while joining.
    // The public contract forbids concurrent executor destruction and search calls.
    void stop_and_join() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        available.notify_all();
        for (std::thread& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    void worker_loop() {
        for (;;) {
            std::packaged_task<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex);
                available.wait(lock, [this] { return stopping || !pending.empty(); });
                if (stopping && pending.empty()) {
                    return;
                }
                task = std::move(pending.front());
                pending.pop_front();
            }
            // packaged_task stores any query exception in its future. Work runs
            // outside the queue lock, so other workers and callers can progress.
            task();
        }
    }

    std::future<void> submit(std::packaged_task<void()> task) {
        std::future<void> completion = task.get_future();
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (stopping) {
                throw std::runtime_error("Cannot submit to a stopping BatchExecutor");
            }
            pending.push_back(std::move(task));
        }
        available.notify_one();
        return completion;
    }

    // Each range owns its query copies/heaps/statistics and writes distinct output
    // elements. O(range length * single-query work), O(dim + per-query scratch).
    void run_range(const Matrix& queries, std::size_t begin, std::size_t end,
                   std::size_t k, BatchMethod method, std::vector<SearchResult>& output) const {
        for (std::size_t i = begin; i < end; ++i) {
            const float* first = queries.row(i);
            const std::vector<float> query(first, first + queries.dim());
            if (method == BatchMethod::Flat) {
                output[i] = search_flat(*data, query, k);
            } else {
                output[i] = tree->search(query, k);
            }
        }
    }
};

// Input: immutable shared data, optional matching tree, T in [1,64]. Output:
// persistent executor; T==1 creates no worker. Startup/storage is O(T).
BatchExecutor::BatchExecutor(std::shared_ptr<const Matrix> data,
                             std::shared_ptr<const KDTree> tree, std::size_t threads)
    : impl_(std::make_unique<Impl>(std::move(data), std::move(tree), threads)) {}

BatchExecutor::~BatchExecutor() = default;

std::size_t BatchExecutor::threads() const noexcept {
    return impl_->thread_count;
}

// Output: one independent result per query, preserving query order. Validation
// precedes empty/k==0 returns. T>1 partitions into min(T,Q) contiguous ranges.
// Work is Q single-query searches + O(Q+T); ideal speedup is not guaranteed.
// Output space O(Q + Q*min(k,N)); future/range state O(min(T,Q)), plus each
// active worker's query vector and the selected serial engine's scratch space.
std::vector<SearchResult> BatchExecutor::search(const Matrix& queries, std::size_t k,
                                               BatchMethod method) {
    if (queries.dim() != impl_->data->dim()) {
        throw std::invalid_argument("Batch query dimension does not match Matrix");
    }
    if (method != BatchMethod::Flat && method != BatchMethod::KDTree) {
        throw std::invalid_argument("Unknown BatchMethod");
    }
    if (method == BatchMethod::KDTree && !impl_->tree) {
        throw std::invalid_argument("KDTree batch method requires a tree");
    }
    std::vector<SearchResult> output(queries.rows());
    if (queries.rows() == 0 || k == 0 || impl_->data->rows() == 0) {
        return output;
    }
    if (impl_->thread_count == 1) {
        impl_->run_range(queries, 0, queries.rows(), k, method, output);
        return output;
    }

    const std::size_t ranges = std::min(impl_->thread_count, queries.rows());
    std::vector<std::future<void>> completions;
    // Reserve before any submission: push_back then moves a future without an
    // allocation or exception that could lose track of an already queued task.
    completions.reserve(ranges);
    std::exception_ptr first_error;
    std::size_t begin = 0;
    try {
        for (std::size_t range = 0; range < ranges; ++range) {
            const std::size_t length = queries.rows() / ranges +
                (range < queries.rows() % ranges ? 1 : 0);
            const std::size_t end = begin + length;
            Impl* const state = impl_.get();
            std::packaged_task<void()> task([state, &queries, &output, begin, end, k, method] {
                state->run_range(queries, begin, end, k, method, output);
            });
            completions.push_back(state->submit(std::move(task)));
            begin = end;
        }
    } catch (...) {
        first_error = std::current_exception();
    }
    for (std::future<void>& completion : completions) {
        try {
            completion.get();
        } catch (...) {
            if (!first_error) {
                first_error = std::current_exception();
            }
        }
    }
    // All captures of output/queries have finished before either is destroyed or
    // the caller regains control, including partial dispatch and failed tasks.
    if (first_error) {
        std::rethrow_exception(first_error);
    }
    return output;
}

} // namespace exactsearch
