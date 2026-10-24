// Windows CRT narrow argv may lose Unicode. Convert the wide entry to UTF-8;
// file code subsequently uses filesystem::u8path, not the current ANSI locale.
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

int exactsearch_program(int argc, char** argv);

#ifdef _WIN32
int wmain(int argc, wchar_t** wide_argv) {
    try {
        std::vector<std::string> storage;
        storage.reserve(argc);
        for (int i=0; i<argc; ++i) {
            const int required=WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                wide_argv[i], -1, nullptr, 0, nullptr, nullptr);
            if (!required) throw std::runtime_error("invalid Unicode command argument");
            std::string value(static_cast<std::size_t>(required), '\0');
            if (!WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide_argv[i], -1,
                                     value.data(), required, nullptr, nullptr))
                throw std::runtime_error("UTF-8 command argument conversion failed");
            value.pop_back();
            storage.push_back(std::move(value));
        }
        std::vector<char*> argv;
        for (auto& value: storage) argv.push_back(value.data());
        argv.push_back(nullptr);
        return exactsearch_program(argc,argv.data());
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
#else
int main(int argc,char** argv) { return exactsearch_program(argc,argv); }
#endif
