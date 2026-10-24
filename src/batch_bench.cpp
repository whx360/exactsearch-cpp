#include "exactsearch/batch.hpp"
#include "exactsearch/io.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using exactsearch::BatchMethod;
using exactsearch::Matrix;
using exactsearch::Neighbor;
using exactsearch::SearchResult;
constexpr std::size_t k = 10;
constexpr std::size_t leaf = 16;
constexpr std::size_t max_base_bytes = 256ULL * 1024 * 1024;
constexpr std::size_t max_query_bytes = 64ULL * 1024 * 1024;

struct Configuration {
    BatchMethod method;
    std::size_t threads;
};
constexpr std::array<Configuration, 8> configurations{{
    {BatchMethod::Flat, 1}, {BatchMethod::Flat, 2},
    {BatchMethod::Flat, 4}, {BatchMethod::Flat, 8},
    {BatchMethod::KDTree, 1}, {BatchMethod::KDTree, 2},
    {BatchMethod::KDTree, 4}, {BatchMethod::KDTree, 8}
}};

const char* method_name(BatchMethod method) {
    return method == BatchMethod::Flat ? "flat" : "kdtree";
}

std::size_t parse_repeats(const std::string& value) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("REPEATS requires an unsigned decimal integer in [1,50]");
    }
    unsigned int parsed = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
        parsed == 0 || parsed > 50) {
        throw std::invalid_argument("REPEATS must be in [1,50]");
    }
    return parsed;
}

void validate_dataset(const std::string& name) {
    if (name.empty() || name.size() > 128 ||
        std::any_of(name.begin(), name.end(), [](unsigned char c) { return c < 32 || c == 127; })) {
        throw std::invalid_argument("DATASET must contain 1..128 printable bytes");
    }
}

std::string csv_string(const std::string& value) {
    std::string output = "\"";
    for (char c : value) {
        if (c == '"') output += '"';
        output += c;
    }
    return output + '"';
}

std::size_t node_count(std::size_t rows) {
    if (rows == 0) return 0;
    if (rows <= leaf) return 1;
    return 1 + node_count(rows / 2) + node_count(rows - rows / 2);
}

// Limits are CLI guards, not RSS measurements. Dataset validation itself is in IO.
void protect_working_set(const Matrix& base, const Matrix& queries) {
    if (base.dim() != queries.dim()) {
        throw std::invalid_argument("Base/query dimensions differ");
    }
    if (base.rows() == 0 || queries.rows() == 0 || base.rows() > 1000000 ||
        queries.rows() > 10000 || base.dim() > 4096) {
        throw std::length_error("Require 1..1000000 base rows, 1..10000 queries, dim<=4096");
    }
    const long double work = static_cast<long double>(base.rows()) * base.dim() * queries.rows();
    if (work > 2000000000.0L) {
        throw std::length_error("n*dim*queries exceeds 2e9 batch-benchmark work limit");
    }
    const long double index_estimate = 2.0L *
        (static_cast<long double>(base.rows()) * sizeof(std::uint32_t) +
         static_cast<long double>(node_count(base.rows())) *
             (32.0L + 2.0L * base.dim() * sizeof(float)));
    const long double result_bytes = static_cast<long double>(queries.rows()) *
        std::min(k, base.rows()) * sizeof(Neighbor);
    const long double working = index_estimate + 2.0L * result_bytes +
        static_cast<long double>(base.values().size() + queries.values().size()) * sizeof(float) +
        static_cast<long double>(base.rows()) * sizeof(Neighbor);
    if (index_estimate > 512.0L * 1024 * 1024 || working > 1024.0L * 1024 * 1024) {
        throw std::length_error("Estimated index/working storage exceeds 512MiB/1GiB limits");
    }
}

std::vector<float> query_row(const Matrix& queries, std::size_t i) {
    const float* first = queries.row(i);
    return std::vector<float>(first, first + queries.dim());
}

void verify(const std::vector<std::vector<Neighbor>>& reference,
            const std::vector<SearchResult>& actual, const Configuration& config) {
    if (reference.size() != actual.size()) {
        throw std::runtime_error("Batch result query count mismatch");
    }
    for (std::size_t i = 0; i < reference.size(); ++i) {
        bool matches = reference[i].size() == actual[i].neighbors.size();
        for (std::size_t j = 0; matches && j < reference[i].size(); ++j) {
            matches = reference[i][j].id == actual[i].neighbors[j].id &&
                      reference[i][j].distance2 == actual[i].neighbors[j].distance2;
        }
        if (!matches) {
            throw std::runtime_error(std::string("Exact batch mismatch: engine=") +
                method_name(config.method) + ", threads=" + std::to_string(config.threads) +
                ", query=" + std::to_string(i) + "; failing row is not emitted as consistent");
        }
    }
}

std::uint64_t checksum(const std::vector<SearchResult>& batch) {
    static_assert(sizeof(double) == sizeof(std::uint64_t), "Checksum requires 64-bit double");
    constexpr std::uint64_t prime = 1099511628211ULL;
    std::uint64_t hash = 14695981039346656037ULL;
    for (std::size_t i = 0; i < batch.size(); ++i) {
        hash = (hash ^ static_cast<std::uint64_t>(i)) * prime;
        hash = (hash ^ static_cast<std::uint64_t>(batch[i].neighbors.size())) * prime;
        for (const Neighbor& value : batch[i].neighbors) {
            std::uint64_t bits = 0;
            std::memcpy(&bits, &value.distance2, sizeof(bits));
            hash = (hash ^ value.id) * prime;
            hash = (hash ^ bits) * prime;
        }
    }
    return hash;
}

void usage() {
    std::cout << "exactsearch_batchbench BASE QUERIES OUT_CSV REPEATS DATASET\n"
              << "Inputs are little-endian fvecs; REPEATS is 1..50. k=10, leaf=16.\n"
              << "Runs flat/kdtree with threads=1,2,4,8, rotating configurations per repeat.\n"
              << "startup_ms measures a new BatchExecutor constructor with data/index already loaded.\n"
              << "After one complete warmup batch, batch_ms measures a complete search() call:\n"
              << "result allocation, submission, work and collection included; construction,\n"
              << "index build, verification, checksum and result destruction excluded.\n"
              << "Guards: base<=256MiB, queries<=64MiB; n<=1000000, queries<=10000, dim<=4096;\n"
              << "n*dim*queries<=2e9, estimated index<=512MiB and working storage<=1GiB.\n";
}

} // namespace

// The shared entry.cpp converts Windows wide arguments to UTF-8 before calling us.
int exactsearch_program(int argc, char** argv) {
    try {
        std::cout.imbue(std::locale::classic());
        std::cout << std::setprecision(17);
        if (argc == 2 && std::string(argv[1]) == "--help") {
            usage();
            return 0;
        }
        if (argc != 6) {
            usage();
            throw std::invalid_argument("Expected BASE QUERIES OUT_CSV REPEATS DATASET");
        }
        const std::string base_file(argv[1]);
        const std::string query_file(argv[2]);
        const std::filesystem::path destination = std::filesystem::u8path(argv[3]);
        const std::size_t repeats = parse_repeats(argv[4]);
        const std::string dataset(argv[5]);
        validate_dataset(dataset);
        if (base_file.empty() || query_file.empty() || destination.empty()) {
            throw std::invalid_argument("Input and output paths must be nonempty");
        }
        if (std::filesystem::exists(destination) &&
            (std::filesystem::equivalent(destination, std::filesystem::u8path(base_file)) ||
             std::filesystem::equivalent(destination, std::filesystem::u8path(query_file)))) {
            throw std::invalid_argument("OUT_CSV must not overwrite an input data file");
        }
        const auto data = std::make_shared<const Matrix>(exactsearch::load_fvecs(base_file, max_base_bytes));
        const Matrix queries = exactsearch::load_fvecs(query_file, max_query_bytes);
        protect_working_set(*data, queries);

        std::cout << "Preparing sort references and immutable KDTree (excluded from timings)...\n";
        std::vector<std::vector<Neighbor>> reference;
        reference.reserve(queries.rows());
        for (std::size_t i = 0; i < queries.rows(); ++i) {
            const auto expected = exactsearch::search_sort(*data, query_row(queries, i), k);
            reference.emplace_back(expected.neighbors.begin(), expected.neighbors.end());
        }
        const auto tree = std::make_shared<const exactsearch::KDTree>(data, leaf);
        if (destination.has_parent_path()) {
            std::filesystem::create_directories(destination.parent_path());
        }
        std::ofstream csv;
        csv.exceptions(std::ios::failbit | std::ios::badbit);
        csv.open(destination, std::ios::out | std::ios::trunc);
        csv.imbue(std::locale::classic());
        csv << std::setprecision(17)
            << "dataset,n,dim,queries,k,leaf,engine,threads,repeat,startup_ms,batch_ms,qps,consistent,checksum\n";
        for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
            for (std::size_t position = 0; position < configurations.size(); ++position) {
                const Configuration config = configurations[(position + repeat) % configurations.size()];
                const auto constructor_begin = Clock::now();
                exactsearch::BatchExecutor executor(data, tree, config.threads);
                const auto constructor_end = Clock::now();
                const double startup_ms = std::chrono::duration<double, std::milli>(
                    constructor_end - constructor_begin).count();
                {
                    const auto warmup = executor.search(queries, k, config.method);
                    verify(reference, warmup, config);
                }
                const auto batch_begin = Clock::now();
                const std::vector<SearchResult> result = executor.search(queries, k, config.method);
                const auto batch_end = Clock::now();
                const double batch_ms = std::chrono::duration<double, std::milli>(batch_end - batch_begin).count();
                verify(reference, result, config);
                const std::uint64_t result_checksum = checksum(result);
                const double qps = batch_ms > 0.0 ?
                    static_cast<double>(queries.rows()) * 1000.0 / batch_ms :
                    std::numeric_limits<double>::quiet_NaN();
                csv << csv_string(dataset) << ',' << data->rows() << ',' << data->dim() << ','
                    << queries.rows() << ',' << k << ',' << leaf << ',' << method_name(config.method)
                    << ',' << config.threads << ',' << repeat + 1 << ',' << startup_ms << ','
                    << batch_ms << ',' << qps << ",true," << result_checksum << '\n';
                csv.flush();
                std::cout << "engine=" << method_name(config.method) << " threads=" << config.threads
                          << " repeat=" << repeat + 1 << " startup_ms=" << startup_ms
                          << " batch_ms=" << batch_ms << " qps=" << qps
                          << " consistent=true checksum=" << result_checksum << '\n';
                // Result destruction and persistent worker shutdown occur after timing.
            }
        }
        csv.close();
        std::cout << "CSV saved: " << argv[3] << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
