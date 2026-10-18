#include <cstdio>
#include <cstring>
#include <iostream>

#include "core/version.hpp"
#include "io/input.hpp"
#include "io/run.hpp"

namespace {

void PrintUsage(const char* argv0) {
  std::fprintf(stderr, "usage: %s [--version | input.in]\n", argv0);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 2) {
    PrintUsage(argv[0]);
    return 2;
  }
  if (argc == 1 || std::strcmp(argv[1], "--version") == 0) {
    std::printf("TileMD %s\n", tilemd::kVersion);
    return 0;
  }
  try {
    const tilemd::SimulationInput input = tilemd::ParseInputFile(argv[1]);
    const std::vector<tilemd::ThermoRow> rows = tilemd::RunReference(input);
    tilemd::WriteThermo(std::cout, rows);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "TileMD: %s\n", error.what());
    return 1;
  }
  return 0;
}
