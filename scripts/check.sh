#!/usr/bin/env bash
# Format-check, configure, build, and run the CPU tests. This is what CI runs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${BUILD_DIR:-build}"

if ! command -v clang-format >/dev/null 2>&1; then
  echo "error: clang-format not found on PATH" >&2
  exit 1
fi

files=()
while IFS= read -r -d '' f; do
  files+=("$f")
done < <(find src tests -type f \
  \( -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.cc' \
     -o -name '*.h' -o -name '*.hpp' \) -print0)

if ((${#files[@]} > 0)); then
  clang-format --dry-run --Werror "${files[@]}"
fi

configure_and_test() {
  cmake -S . -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DTILEMD_BUILD_TESTS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  cmake --build "$BUILD_DIR" --parallel
  ctest --test-dir "$BUILD_DIR" -L cpu --output-on-failure
}

# The Visual Studio generator requires the CUDA MSBuild toolset, which is not
# always registered. Ninja drives nvcc directly. On Windows that still needs
# the MSVC environment (INCLUDE/LIB), so load vcvars64.bat first.
# Run a .cmd file. Git Bash mangles quoted Windows paths passed to cmd.exe,
# so discovery and the build are written into a batch file and executed by path.
run_cmd_file() {
  # Git Bash treats an env value that starts with "/" as a POSIX path. CL=/wd4819
  # would become "C:/Program Files/Git/wd4819" and cl.exe would try to link it.
  MSYS2_ENV_CONV_EXCL=CL cmd.exe //c "$(/usr/bin/cygpath -w "$1")"
}

windows_configure_and_test() {
  local vswhere vs_install vcvars ninja nvcc discover cmdfile
  vswhere="/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
  if [[ ! -f "$vswhere" ]]; then
    echo "error: vswhere.exe not found" >&2
    exit 1
  fi
  discover="/tmp/tilemd-vswhere-$$.cmd"
  {
    printf '%s\r\n' "@echo off"
    printf '%s\r\n' "\"$(/usr/bin/cygpath -w "$vswhere")\" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath"
  } >"$discover"
  vs_install="$(run_cmd_file "$discover" | tr -d '\r')"
  rm -f "$discover"
  if [[ -z "$vs_install" ]]; then
    echo "error: Visual Studio C++ toolset not found" >&2
    exit 1
  fi
  vcvars="$(/usr/bin/cygpath -u "$vs_install")/VC/Auxiliary/Build/vcvars64.bat"
  ninja="$(/usr/bin/cygpath -u "$vs_install")/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe"
  if [[ ! -f "$vcvars" ]]; then
    echo "error: vcvars64.bat not found at $vcvars" >&2
    exit 1
  fi
  if [[ ! -f "$ninja" ]]; then
    if ! command -v ninja >/dev/null 2>&1; then
      echo "error: ninja not found" >&2
      exit 1
    fi
    ninja="$(command -v ninja)"
  fi
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "error: nvcc not found on PATH" >&2
    exit 1
  fi
  nvcc="$(command -v nvcc)"

  cmdfile="/tmp/tilemd-check-$$.cmd"
  {
    printf '%s\r\n' "@echo off"
    printf '%s\r\n' "call \"$(/usr/bin/cygpath -w "$vcvars")\" || exit /b 1"
    printf '%s\r\n' "set \"PATH=$(/usr/bin/cygpath -w "$(dirname "$ninja")");%PATH%\""
    printf '%s\r\n' "cd /d \"$(/usr/bin/cygpath -w "$ROOT")\" || exit /b 1"
    printf '%s\r\n' "cmake -S . -B \"$BUILD_DIR\" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=\"$(/usr/bin/cygpath -w "$nvcc")\" -DCMAKE_MAKE_PROGRAM=\"$(/usr/bin/cygpath -w "$ninja")\" -DTILEMD_BUILD_TESTS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON || exit /b 1"
    printf '%s\r\n' "cmake --build \"$BUILD_DIR\" --parallel || exit /b 1"
    printf '%s\r\n' "ctest --test-dir \"$BUILD_DIR\" -L cpu --output-on-failure || exit /b 1"
  } >"$cmdfile"
  local status=0
  run_cmd_file "$cmdfile" || status=$?
  if [[ "$status" -eq 0 ]]; then
    rm -f "$cmdfile"
  else
    echo "error: windows build failed; helper left at $cmdfile" >&2
  fi
  return "$status"
}

uname_s="$(uname -s)"
if [[ "$uname_s" == MINGW* || "$uname_s" == MSYS* || "$uname_s" == CYGWIN* ]]; then
  windows_configure_and_test
else
  configure_and_test
fi
