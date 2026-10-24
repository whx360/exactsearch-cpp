#include "exactsearch/io.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void write_bytes(const std::string& path, const std::vector<unsigned char>& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("cannot create IO test fixture");
}
template<class F> void rejects(F f, const char* message) {
    bool threw = false;
    try { f(); } catch (const std::exception&) { threw = true; }
    require(threw, message);
}
}

void run_io_batch_tests() {
    using namespace exactsearch;
    const std::string path = "build/io_test_fixture.fvecs";
    // Guard cleanup even if a validation assertion throws.
    struct Cleanup { std::string path; ~Cleanup() { std::remove(path.c_str()); } } cleanup{path};
    Matrix original(3, 2, {0.0f, -0.0f, 1.25f, -2.5f, 16777216.0f, 0.5f});
    save_fvecs(original, path);
    auto loaded = load_fvecs(path);
    require(loaded.rows() == 3 && loaded.dim() == 2, "IO shape roundtrip");
    require(loaded.values() == original.values(), "IO values roundtrip");
    require(std::signbit(loaded.row(0)[1]), "IO preserves negative zero");
    rejects([&] { load_fvecs(path, 4); }, "IO allocation cap");
    write_bytes(path, {});
    rejects([&] { load_fvecs(path); }, "empty file rejected");
    write_bytes(path, {0, 0, 0, 0});
    rejects([&] { load_fvecs(path); }, "zero dimension rejected");
    write_bytes(path, {1, 0, 0, 0, 0, 0, 128});
    rejects([&] { load_fvecs(path); }, "truncated float rejected");
    write_bytes(path, {1,0,0,0,0,0,128,63,2,0,0,0,0,0,0,0});
    rejects([&] { load_fvecs(path); }, "inconsistent dimension rejected");
    write_bytes(path, {255,255,255,127});
    rejects([&] { load_fvecs(path); }, "hostile dimension rejected before allocation");
    write_bytes(path, {1,0,0,0,0,0,128,127});
    rejects([&] { load_fvecs(path); }, "infinite value rejected");
    write_bytes(path, {1,0,0,0,0,0,192,127});
    rejects([&] { load_fvecs(path); }, "NaN value rejected");
    rejects([&] { load_fvecs("build/missing_subdir/no_file.fvecs"); }, "missing file rejected");
    rejects([&] { save_fvecs(Matrix(0,2,{}), path); }, "empty fvecs serialization rejected");
    std::cout << "[PASS] strict fvecs IO roundtrip and malformed-input cases\n";
}
