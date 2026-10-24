#include "exactsearch/search.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>

// Phase microbenchmarks: explanatory timing, not hardware-counter sampling.
// The distance phase includes writing an Nd-independent Q*N distance buffer;
// selection phases consume that buffer. Their sums need not equal a fused query.
int exactsearch_program(int argc, char** argv) {
    using namespace exactsearch;
    using Clock = std::chrono::steady_clock;
    try {
        const std::string path = argc > 1 ? argv[1] : "results/profile.csv";
        std::ofstream out(std::filesystem::u8path(path));
        if (!out) throw std::runtime_error("cannot open profile CSV");
        out << "n,dim,queries,k,repeat,distance_write_ms,sort_select_ms,heap_select_ms,checksum\n";
        out << std::setprecision(12);
        constexpr std::size_t n=10000, qcount=128, k=10;
        double live_checksum=0;
        for (std::size_t dim : {4u,16u,64u,128u}) {
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> dist(-1,1);
            std::vector<float> values(n*dim), queries(qcount*dim);
            for (auto& x: values) x=dist(rng);
            for (auto& x: queries) x=dist(rng);
            Matrix data(n,dim,std::move(values));
            std::vector<double> distances(n*qcount);
            for (int repeat=0; repeat<6; ++repeat) {
                const auto start=Clock::now();
                for (std::size_t q=0; q<qcount; ++q)
                    for (std::size_t i=0; i<n; ++i)
                        distances[q*n+i]=squared_l2(data.row(i),queries.data()+q*dim,dim);
                const auto distance_end=Clock::now();
                double checksum=0;
                for (double d: distances) checksum+=d;
                std::vector<std::vector<Neighbor>> sorted(qcount), heaped(qcount);
                const auto sort_start=Clock::now();
                for (std::size_t q=0; q<qcount; ++q) {
                    auto& candidates=sorted[q];
                    candidates.reserve(n);
                    for (std::size_t i=0; i<n; ++i)
                        candidates.push_back({static_cast<std::uint32_t>(i),distances[q*n+i]});
                    std::sort(candidates.begin(),candidates.end(),better);
                    candidates.resize(k);
                }
                const auto sort_end=Clock::now();
                const auto heap_start=Clock::now();
                for (std::size_t q=0; q<qcount; ++q) {
                    BoundedTopK top(k);
                    for (std::size_t i=0; i<n; ++i)
                        top.push({static_cast<std::uint32_t>(i),distances[q*n+i]});
                    heaped[q]=top.sorted();
                }
                const auto heap_end=Clock::now();
                for (std::size_t q=0; q<qcount; ++q)
                    for (std::size_t j=0; j<k; ++j) {
                        if (sorted[q][j].id!=heaped[q][j].id ||
                            sorted[q][j].distance2!=heaped[q][j].distance2)
                            throw std::runtime_error("phase microbenchmark selection mismatch");
                        checksum+=heaped[q][j].distance2+heaped[q][j].id;
                    }
                live_checksum+=checksum;
                if(repeat) out << n << ',' << dim << ',' << qcount << ',' << k << ',' << repeat << ','
                    << std::chrono::duration<double,std::milli>(distance_end-start).count() << ','
                    << std::chrono::duration<double,std::milli>(sort_end-sort_start).count() << ','
                    << std::chrono::duration<double,std::milli>(heap_end-heap_start).count() << ',' << checksum << '\n';
            }
            std::cout << "Profile dim=" << dim << " done\n";
        }
        out.close();
        if(!out) throw std::runtime_error("profile CSV write failed");
        std::cout << "Phase profile checksum=" << live_checksum << '\n';
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
