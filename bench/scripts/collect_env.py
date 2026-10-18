#!/usr/bin/env python3
"""Record the benchmark environment as JSON.

Hostnames are stored only as a SHA-256 digest (PLAN.md section 9.4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
from typing import Any


def run(cmd: list[str]) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)
    return completed.returncode, completed.stdout, completed.stderr


def nvidia_query() -> dict[str, Any]:
    fields = [
        "name",
        "driver_version",
        "compute_cap",
        "ecc.mode.current",
        "clocks.sm",
        "clocks.max.sm",
    ]
    code, out, err = run(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"]
    )
    gpus: list[dict[str, str]] = []
    if code == 0:
        for line in out.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != len(fields):
                continue
            gpus.append(dict(zip(fields, parts)))
    clock_code, clock_out, clock_err = run(["nvidia-smi", "-q", "-d", "CLOCK"])
    return {
        "gpus": gpus,
        "query_error": None if code == 0 else err.strip(),
        "clocks_query": clock_out if clock_code == 0 else "",
        "clocks_query_error": None if clock_code == 0 else clock_err.strip(),
    }


def cuda_version() -> str | None:
    code, out, _err = run(["nvcc", "--version"])
    if code != 0:
        return None
    for line in out.splitlines():
        if "release" in line:
            return line.strip()
    return out.strip() or None


def git_commit() -> str | None:
    code, out, _err = run(["git", "rev-parse", "HEAD"])
    if code != 0:
        return None
    digest = out.strip()
    return digest or None


def cmake_flags(cache_path: str | None) -> dict[str, str]:
    if not cache_path:
        return {}
    wanted = {
        "CMAKE_BUILD_TYPE",
        "CMAKE_CUDA_ARCHITECTURES",
        "CMAKE_CUDA_FLAGS",
        "CMAKE_CUDA_FLAGS_RELEASE",
        "CMAKE_CXX_FLAGS",
        "CMAKE_CXX_FLAGS_RELEASE",
        "TILEMD_ARCHS",
        "TILEMD_FAST_MATH",
    }
    found: dict[str, str] = {}
    try:
        with open(cache_path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if (
                    ":" not in line
                    or "=" not in line
                    or line.startswith("#")
                    or line.startswith("//")
                ):
                    continue
                key, value = line.split("=", 1)
                name = key.split(":", 1)[0]
                if name in wanted:
                    found[name] = value.strip()
    except OSError as exc:
        found["error"] = str(exc)
    return found


def build_record(cache_path: str | None) -> dict[str, Any]:
    gpu = nvidia_query()
    first = gpu["gpus"][0] if gpu["gpus"] else {}
    hostname = socket.gethostname().encode("utf-8", errors="replace")
    return {
        "gpu_name": first.get("name"),
        "driver": first.get("driver_version"),
        "cuda_version": cuda_version(),
        "compute_capability": first.get("compute_cap"),
        "ecc": first.get("ecc.mode.current"),
        "sm_clocks_mhz": first.get("clocks.sm"),
        "sm_clocks_max_mhz": first.get("clocks.max.sm"),
        "gpus": gpu["gpus"],
        "clocks_query": gpu["clocks_query"],
        "cpu": platform.processor() or platform.machine(),
        "hostname_sha256": hashlib.sha256(hostname).hexdigest(),
        "git_commit": git_commit(),
        "build_flags": cmake_flags(cache_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        help="Write JSON to this path as well as stdout",
    )
    parser.add_argument(
        "--cmake-cache",
        help="Path to CMakeCache.txt; selected flags are copied into build_flags",
    )
    args = parser.parse_args()
    record = build_record(args.cmake_cache)
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    sys.stdout.write(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
