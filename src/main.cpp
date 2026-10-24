#include "exactsearch/io.hpp"
#include "exactsearch/search.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Options = std::map<std::string, std::string>;
using exactsearch::Matrix;
using exactsearch::Neighbor;
using exactsearch::SearchResult;
constexpr std::size_t max_rows = 1000000;
constexpr std::size_t max_dim = 4096;
constexpr std::size_t max_queries = 10000;
constexpr std::size_t max_base_bytes = 256ULL * 1024 * 1024;
constexpr std::size_t max_query_bytes = 64ULL * 1024 * 1024;
constexpr std::size_t max_reference_items = 4000000;

enum class Engine { sort, flat, kdtree };
constexpr std::array<Engine, 3> engines{Engine::sort, Engine::flat, Engine::kdtree};

const char* engine_name(Engine engine) {
    switch (engine) {
    case Engine::sort: return "sort";
    case Engine::flat: return "flat";
    case Engine::kdtree: return "kdtree";
    }
    throw std::logic_error("Unknown internal engine");
}

void usage() {
    std::cout
        << "ExactSearch: exact squared-L2 top-k, ordered by (distance2,id).\n\n"
        << "  exactsearch demo\n"
        << "  exactsearch query --base FILE --queries FILE [--query-index N]\n"
        << "       [--k K] [--engine sort|flat|kdtree] [--leaf L]\n"
        << "  exactsearch bench --out CSV [--base FILE --queries FILE |\n"
        << "       --n N --dim D --queries-count Q --seed S]\n"
        << "       [--k K] [--leaf L] [--repeats R] [--dataset NAME]\n"
        << "       [--limit-queries Q (file inputs)] [--engines sort,flat,kdtree]\n\n"
        << "Defaults: query-index=0, k=10, engine=flat, leaf=16, repeats=3;\n"
        << "synthetic n=10000, dim=8, queries-count=100, seed=42.\n"
        << "File inputs are strict little-endian fvecs. Synthetic values use\n"
        << "mt19937 and uniform_real_distribution<float>(-1,1); base then queries.\n"
        << "Benchmark compares every query to precomputed sort results.\n"
        << "Each repeat rotates engine order; each engine batch is warmed up.\n"
        << "Latency measures only a search call through return; result destruction,\n"
        << "verification, checksum, I/O and tree build are excluded. batch_ms is\n"
        << "the sum of query call times; p50/p95 use nearest-rank percentiles.\n"
        << "Index bytes estimate allocated index capacity, excluding shared data.\n"
        << "Limits: n<=1000000, dim<=4096, queries<=10000, repeats<=50;\n"
        << "base<=256MiB, queries<=64MiB, reference<=4000000 neighbors,\n"
        << "estimated index<=512MiB and total working storage<=1GiB;\n"
        << "n*dim*queries<=2e9 per engine/repeat. Bench n and queries must be >0.\n";
}

// Parse each option exactly once. All options take one value; unknown flags fail.
Options parse_options(int argc, char** argv, const std::set<std::string>& allowed) {
    Options options;
    for (int i = 2; i < argc; i += 2) {
        const std::string key(argv[i]);
        if (allowed.find(key) == allowed.end()) {
            throw std::invalid_argument("Unknown option: " + key);
        }
        if (options.find(key) != options.end()) {
            throw std::invalid_argument("Duplicate option: " + key);
        }
        if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0 ||
            std::string(argv[i + 1]).empty()) {
            throw std::invalid_argument("Missing value for " + key);
        }
        options.emplace(key, argv[i + 1]);
    }
    return options;
}

std::string required(const Options& options, const std::string& key) {
    const auto found = options.find(key);
    if (found == options.end()) {
        throw std::invalid_argument("Required option missing: " + key);
    }
    return found->second;
}

std::string optional(const Options& options, const std::string& key,
                     const std::string& fallback) {
    const auto found = options.find(key);
    return found == options.end() ? fallback : found->second;
}

// Strict decimal unsigned parsing rejects signs, trailing text and overflow. O(len).
std::uint64_t unsigned_number(const std::string& text, const std::string& key) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument(key + " requires an unsigned decimal integer");
    }
    std::uint64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
        throw std::invalid_argument(key + " integer is out of range");
    }
    return value;
}

std::size_t number(const Options& options, const std::string& key,
                   std::size_t fallback, std::size_t minimum, std::size_t maximum) {
    const std::uint64_t value = unsigned_number(optional(options, key, std::to_string(fallback)), key);
    if (value < minimum || value > maximum ||
        value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(key + " must be in [" + std::to_string(minimum) +
                                    "," + std::to_string(maximum) + "]");
    }
    return static_cast<std::size_t>(value);
}

Engine parse_engine(const std::string& name) {
    for (Engine engine : engines) {
        if (name == engine_name(engine)) {
            return engine;
        }
    }
    throw std::invalid_argument("--engine must be sort, flat or kdtree");
}

std::vector<Engine> parse_engines(const std::string& names) {
    std::vector<Engine> selected;
    std::size_t begin = 0;
    while (begin <= names.size()) {
        const std::size_t comma = names.find(',', begin);
        const std::string name = names.substr(begin, comma == std::string::npos ? comma : comma - begin);
        const Engine engine = parse_engine(name);
        if (std::find(selected.begin(), selected.end(), engine) != selected.end()) {
            throw std::invalid_argument("--engines contains a duplicate engine");
        }
        selected.push_back(engine);
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return selected;
}

void check_shape(std::size_t rows, std::size_t dim, std::size_t row_limit,
                 std::size_t byte_limit, const char* label) {
    if (rows > row_limit || dim == 0 || dim > max_dim ||
        rows > byte_limit / sizeof(float) / dim) {
        throw std::length_error(std::string(label) + " exceeds CLI shape/memory limits");
    }
}

std::shared_ptr<const Matrix> load_base(const std::string& path) {
    auto matrix = std::make_shared<const Matrix>(exactsearch::load_fvecs(path, max_base_bytes));
    check_shape(matrix->rows(), matrix->dim(), max_rows, max_base_bytes, "Base data");
    return matrix;
}

Matrix load_queries(const std::string& path, std::size_t dimension) {
    Matrix matrix = exactsearch::load_fvecs(path, max_query_bytes);
    check_shape(matrix.rows(), matrix.dim(), max_queries, max_query_bytes, "Query data");
    if (matrix.dim() != dimension) {
        throw std::invalid_argument("Base/query dimensions differ");
    }
    return matrix;
}

// Median splits bound recursion depth independently of coordinate duplicates.
std::size_t estimated_nodes(std::size_t rows, std::size_t leaf) {
    if (rows == 0) return 0;
    if (rows <= leaf) return 1;
    return 1 + estimated_nodes(rows / 2, leaf) + estimated_nodes(rows - rows / 2, leaf);
}

// Conservative storage guard, not a measured RSS figure. O(number of tree nodes).
void check_working_storage(const Matrix& data, const Matrix& queries,
                           std::size_t k, std::size_t leaf, bool include_reference,
                           bool include_index = true) {
    const std::size_t count = std::min(k, data.rows());
    if (include_reference && count > max_reference_items / std::max<std::size_t>(1, queries.rows())) {
        throw std::length_error("Reference result cache exceeds 4000000 neighbors");
    }
    const long double node_count = include_index ?
        static_cast<long double>(estimated_nodes(data.rows(), leaf)) : 0.0L;
    // Allow 2x logical storage for ordinary vector capacity growth.
    const long double index_estimate = include_index ? 2.0L *
        (static_cast<long double>(data.rows()) * sizeof(std::uint32_t) +
         node_count * (32.0L + 2.0L * data.dim() * sizeof(float))) : 0.0L;
    const long double reference = include_reference ?
        static_cast<long double>(queries.rows()) * count * sizeof(Neighbor) : 0.0L;
    const long double working = index_estimate + reference +
        static_cast<long double>(data.values().size() + queries.values().size()) * sizeof(float) +
        static_cast<long double>(data.rows()) * sizeof(Neighbor);
    if (index_estimate > 512.0L * 1024 * 1024 || working > 1024.0L * 1024 * 1024) {
        throw std::length_error("Estimated index/working storage exceeds CLI protection limits");
    }
}

std::vector<float> query_row(const Matrix& queries, std::size_t row) {
    const float* begin = queries.row(row);
    return std::vector<float>(begin, begin + queries.dim());
}

SearchResult run_search(Engine engine, const Matrix& data, const exactsearch::KDTree* tree,
                        const std::vector<float>& query, std::size_t k) {
    switch (engine) {
    case Engine::sort: return exactsearch::search_sort(data, query, k);
    case Engine::flat: return exactsearch::search_flat(data, query, k);
    case Engine::kdtree:
        if (!tree) throw std::logic_error("KDTree engine has no index");
        return tree->search(query, k);
    }
    throw std::logic_error("Unknown internal engine");
}

void verify(const std::vector<Neighbor>& expected, const std::vector<Neighbor>& actual,
            Engine engine, std::size_t query_index) {
    bool matches = expected.size() == actual.size();
    for (std::size_t i = 0; matches && i < expected.size(); ++i) {
        matches = expected[i].id == actual[i].id &&
                  expected[i].distance2 == actual[i].distance2;
    }
    if (!matches) {
        throw std::runtime_error(std::string("Exact result mismatch: engine=") +
                                 engine_name(engine) + ", query=" + std::to_string(query_index) +
                                 "; failing batch is not emitted as consistent");
    }
}

void checksum_result(std::uint64_t& hash, const std::vector<Neighbor>& values,
                     std::size_t query_index) {
    static_assert(sizeof(double) == sizeof(std::uint64_t), "Checksum requires 64-bit double");
    constexpr std::uint64_t prime = 1099511628211ULL;
    hash = (hash ^ static_cast<std::uint64_t>(query_index)) * prime;
    hash = (hash ^ static_cast<std::uint64_t>(values.size())) * prime;
    for (const Neighbor& value : values) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value.distance2, sizeof(bits));
        hash = (hash ^ value.id) * prime;
        hash = (hash ^ bits) * prime;
    }
}

// Nearest-rank percentile: sorted[ceil(p*n)-1]. O(1) once latency is sorted.
double percentile(const std::vector<double>& sorted, double p) {
    const std::size_t rank = static_cast<std::size_t>(std::ceil(p * sorted.size()));
    return sorted[std::max<std::size_t>(1, rank) - 1];
}

std::string csv_string(const std::string& value) {
    std::string escaped = "\"";
    for (char c : value) {
        if (c == '"') escaped += '"';
        escaped += c;
    }
    return escaped + '"';
}

void check_dataset_name(const std::string& name) {
    if (name.empty() || name.size() > 128 ||
        std::any_of(name.begin(), name.end(), [](unsigned char c) { return c < 32 || c == 127; })) {
        throw std::invalid_argument("--dataset must contain 1..128 printable bytes");
    }
}

void demo() {
    auto data = std::make_shared<const Matrix>(4, 2,
        std::vector<float>{0, 0, 1, 0, 0, 1, 2, 2});
    exactsearch::KDTree tree(data, 1);
    const std::vector<float> query{0, 0};
    const auto expected = exactsearch::search_sort(*data, query, 3).neighbors;
    std::cout << "Points: id0=(0,0), id1=(1,0), id2=(0,1), id3=(2,2)\n"
              << "Query=(0,0), k=3. Expected (id,distance2): (0,0) (1,1) (2,1)\n";
    for (Engine engine : engines) {
        const auto result = run_search(engine, *data, &tree, query, 3);
        verify(expected, result.neighbors, engine, 0);
        std::cout << engine_name(engine) << ':';
        for (const Neighbor& value : result.neighbors) {
            std::cout << " (" << value.id << ',' << value.distance2 << ')';
        }
        std::cout << " distance_evaluations=" << result.stats.distance_evaluations << '\n';
    }
}

void query_command(const Options& options) {
    const std::size_t k = number(options, "--k", 10, 0, max_rows);
    const std::size_t leaf = number(options, "--leaf", 16, 1, max_rows);
    const std::size_t index = number(options, "--query-index", 0, 0, max_queries - 1);
    const Engine engine = parse_engine(optional(options, "--engine", "flat"));
    const auto data = load_base(required(options, "--base"));
    const Matrix queries = load_queries(required(options, "--queries"), data->dim());
    if (index >= queries.rows()) throw std::out_of_range("--query-index is outside query data");
    std::unique_ptr<exactsearch::KDTree> tree;
    if (engine == Engine::kdtree) {
        check_working_storage(*data, queries, k, leaf, false);
        tree = std::make_unique<exactsearch::KDTree>(data, leaf);
    }
    const auto result = run_search(engine, *data, tree.get(), query_row(queries, index), k);
    std::cout << "engine=" << engine_name(engine) << " n=" << data->rows()
              << " dim=" << data->dim() << " query_index=" << index
              << " returned=" << result.neighbors.size() << "\nid,distance2\n";
    for (const Neighbor& value : result.neighbors) {
        std::cout << value.id << ',' << value.distance2 << '\n';
    }
    std::cout << "distance_evaluations=" << result.stats.distance_evaluations
              << " nodes_visited=" << result.stats.nodes_visited
              << " box_evaluations=" << result.stats.box_evaluations
              << " heap_updates=" << result.stats.heap_updates << '\n';
}

// Deterministic within a given standard-library implementation; mt19937 is fixed,
// but uniform_real_distribution conversion is not specified across all libraries.
Matrix synthetic_matrix(std::size_t rows, std::size_t dim, std::mt19937& generator) {
    std::vector<float> values(rows * dim);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    for (float& value : values) value = distribution(generator);
    return Matrix(rows, dim, std::move(values));
}

void bench_command(const Options& options) {
    const std::size_t k = number(options, "--k", 10, 0, max_rows);
    const std::size_t leaf = number(options, "--leaf", 16, 1, max_rows);
    const std::size_t repeats = number(options, "--repeats", 3, 1, 50);
    const std::string output_path = required(options, "--out");
    const std::vector<Engine> selected_engines =
        parse_engines(optional(options, "--engines", "sort,flat,kdtree"));
    const bool needs_tree = std::find(selected_engines.begin(), selected_engines.end(),
                                      Engine::kdtree) != selected_engines.end();
    const bool file_mode = options.count("--base") != 0 || options.count("--queries") != 0;
    const std::string dataset = optional(options, "--dataset", file_mode ? "fvecs" : "synthetic_uniform");
    check_dataset_name(dataset);
    std::shared_ptr<const Matrix> data;
    std::unique_ptr<Matrix> queries;
    if (file_mode) {
        for (const char* key : {"--n", "--dim", "--queries-count", "--seed"}) {
            if (options.count(key)) {
                throw std::invalid_argument(std::string(key) + " cannot be combined with file inputs");
            }
        }
        data = load_base(required(options, "--base"));
        queries = std::make_unique<Matrix>(load_queries(required(options, "--queries"), data->dim()));
        const std::size_t limit = number(options, "--limit-queries", max_queries, 1, max_queries);
        if (limit < queries->rows()) {
            const auto end = queries->values().begin() +
                static_cast<std::ptrdiff_t>(limit * queries->dim());
            queries = std::make_unique<Matrix>(limit, queries->dim(),
                std::vector<float>(queries->values().begin(), end));
        }
    } else {
        if (options.count("--limit-queries")) {
            throw std::invalid_argument("--limit-queries requires file inputs; use --queries-count for synthetic data");
        }
        const std::size_t n = number(options, "--n", 10000, 1, max_rows);
        const std::size_t dim = number(options, "--dim", 8, 1, max_dim);
        const std::size_t count = number(options, "--queries-count", 100, 1, max_queries);
        const auto seed = static_cast<std::uint32_t>(number(options, "--seed", 42, 0,
            std::numeric_limits<std::uint32_t>::max()));
        check_shape(n, dim, max_rows, max_base_bytes, "Synthetic base");
        check_shape(count, dim, max_queries, max_query_bytes, "Synthetic queries");
        std::mt19937 generator(seed);
        data = std::make_shared<const Matrix>(synthetic_matrix(n, dim, generator));
        queries = std::make_unique<Matrix>(synthetic_matrix(count, dim, generator));
        std::cout << "synthetic_seed=" << seed << '\n';
    }
    if (data->rows() == 0 || queries->rows() == 0) {
        throw std::invalid_argument("Benchmark requires nonempty base and query data");
    }
    const long double coordinate_work = static_cast<long double>(data->rows()) *
        data->dim() * queries->rows();
    if (coordinate_work > 2000000000.0L) {
        throw std::length_error("n*dim*queries exceeds 2e9 CLI work protection limit");
    }
    check_working_storage(*data, *queries, k, leaf, true, needs_tree);

    std::cout << "Preparing exact reference results (excluded from benchmark timing)...\n";
    std::vector<std::vector<Neighbor>> reference;
    reference.reserve(queries->rows());
    for (std::size_t i = 0; i < queries->rows(); ++i) {
        const auto result = exactsearch::search_sort(*data, query_row(*queries, i), k);
        // Copy only the returned elements, rather than retaining sort's N capacity.
        reference.emplace_back(result.neighbors.begin(), result.neighbors.end());
    }
    std::unique_ptr<exactsearch::KDTree> tree;
    double build_ms = 0.0;
    if (needs_tree) {
        const auto build_begin = Clock::now();
        tree = std::make_unique<exactsearch::KDTree>(data, leaf);
        build_ms = std::chrono::duration<double, std::milli>(Clock::now() - build_begin).count();
    }

    const auto destination = std::filesystem::u8path(output_path);
    if (file_mode && std::filesystem::exists(destination) &&
        (std::filesystem::equivalent(destination, std::filesystem::u8path(required(options, "--base"))) ||
         std::filesystem::equivalent(destination, std::filesystem::u8path(required(options, "--queries"))))) {
        throw std::invalid_argument("--out must not overwrite an input data file");
    }
    if (destination.has_parent_path()) std::filesystem::create_directories(destination.parent_path());
    std::ofstream csv;
    csv.exceptions(std::ios::failbit | std::ios::badbit);
    csv.open(destination, std::ios::out | std::ios::trunc);
    csv.imbue(std::locale::classic());
    csv << std::setprecision(17)
        << "dataset,n,dim,queries,k,leaf,engine,threads,repeat,p50_us,p95_us,batch_ms,qps,"
        << "mean_distance_evals,mean_nodes,mean_box_evals,mean_heap_updates,index_build_ms,"
        << "index_bytes,data_bytes,consistent,checksum\n";
    std::cout << "Timing: batch_ms=sum(search-call durations); verification/checksum/destruction excluded.\n";
    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
        for (std::size_t position = 0; position < selected_engines.size(); ++position) {
            const Engine engine = selected_engines[(position + repeat) % selected_engines.size()];
            for (std::size_t i = 0; i < std::min<std::size_t>(3, queries->rows()); ++i) {
                const auto warmup = run_search(engine, *data, tree.get(), query_row(*queries, i), k);
                verify(reference[i], warmup.neighbors, engine, i);
            }
            std::vector<double> latency;
            latency.reserve(queries->rows());
            double total_us = 0.0;
            long double distance_sum = 0.0L, node_sum = 0.0L, box_sum = 0.0L, update_sum = 0.0L;
            std::uint64_t checksum = 14695981039346656037ULL;
            for (std::size_t i = 0; i < queries->rows(); ++i) {
                const std::vector<float> query = query_row(*queries, i);
                const auto start = Clock::now();
                const SearchResult result = run_search(engine, *data, tree.get(), query, k);
                const auto finish = Clock::now();
                const double elapsed = std::chrono::duration<double, std::micro>(finish - start).count();
                latency.push_back(elapsed);
                total_us += elapsed;
                verify(reference[i], result.neighbors, engine, i);
                checksum_result(checksum, result.neighbors, i);
                distance_sum += result.stats.distance_evaluations;
                node_sum += result.stats.nodes_visited;
                box_sum += result.stats.box_evaluations;
                update_sum += result.stats.heap_updates;
            }
            std::sort(latency.begin(), latency.end());
            const double p50 = percentile(latency, 0.50);
            const double p95 = percentile(latency, 0.95);
            const double batch_ms = total_us / 1000.0;
            const double qps = total_us > 0.0 ? queries->rows() * 1000000.0 / total_us :
                std::numeric_limits<double>::quiet_NaN();
            const long double count = static_cast<long double>(queries->rows());
            csv << csv_string(dataset) << ',' << data->rows() << ',' << data->dim() << ','
                << queries->rows() << ',' << k << ',' << leaf << ',' << engine_name(engine)
                << ",1," << repeat + 1 << ',' << p50 << ',' << p95 << ',' << batch_ms << ',' << qps
                << ',' << distance_sum / count << ',' << node_sum / count << ',' << box_sum / count
                << ',' << update_sum / count << ',' << (engine == Engine::kdtree ? build_ms : 0.0)
                << ',' << (engine == Engine::kdtree ? tree->index_bytes() : 0)
                << ',' << data->values().size() * sizeof(float) << ",true," << checksum << '\n';
            csv.flush();
            std::cout << "engine=" << engine_name(engine) << " repeat=" << repeat + 1
                      << " p50_us=" << p50 << " p95_us=" << p95 << " batch_ms=" << batch_ms
                      << " qps=" << qps << " consistent=true checksum=" << checksum << '\n';
        }
    }
    csv.close();
    std::cout << "CSV saved: " << output_path << '\n';
}

} // namespace

int exactsearch_program(int argc, char** argv) {
    try {
        std::cout.imbue(std::locale::classic());
        std::cout << std::setprecision(17);
        if (argc == 1 || (argc == 2 && (std::string(argv[1]) == "--help" ||
                                      std::string(argv[1]) == "help")) ||
            (argc == 3 && std::string(argv[2]) == "--help" &&
             (std::string(argv[1]) == "demo" || std::string(argv[1]) == "query" ||
              std::string(argv[1]) == "bench"))) {
            usage();
            return 0;
        }
        const std::string command(argv[1]);
        if (command == "demo") {
            if (argc != 2) throw std::invalid_argument("demo accepts no options");
            demo();
        } else if (command == "query") {
            query_command(parse_options(argc, argv,
                {"--base", "--queries", "--query-index", "--k", "--engine", "--leaf"}));
        } else if (command == "bench") {
            bench_command(parse_options(argc, argv,
                {"--base", "--queries", "--n", "--dim", "--queries-count", "--seed", "--k",
                 "--leaf", "--repeats", "--out", "--dataset", "--limit-queries", "--engines"}));
        } else {
            throw std::invalid_argument("Unknown command: " + command);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
