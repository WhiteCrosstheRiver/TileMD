#include <cstdio>
#include <cstring>

#include "core/version.hpp"

namespace {

void PrintUsage(const char* argv0) { std::fprintf(stderr, "usage: %s [--version]\n", argv0); }

}  // namespace

int main(int argc, char** argv) {
  if (argc > 2) {
    PrintUsage(argv[0]);
    return 2;
  }
  if (argc == 2 && std::strcmp(argv[1], "--version") != 0) {
    PrintUsage(argv[0]);
    return 2;
  }
  std::printf("TileMD %s\n", tilemd::kVersion);
  return 0;
}
