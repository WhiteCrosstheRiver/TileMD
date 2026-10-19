#!/usr/bin/env python3
"""Time existing TileMD paths with the PLAN.md section 8 step counts.

One JSON file per path and size. The cluster-tile Run is the fused leapfrog,
so there is no separate tile integrator on this tree.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = {32000: 100_000, 256000: 20_000, 2_048_000: 5_000}
PATHS = ("atom", "fused-tile", "atom-ref")


def median(values: list[float]) -> float:
    return statistics.median(values)


def run_once(binary: Path, path: str, n: int, steps: int, warmup: int, dump: str) -> dict:
    cmd = [
        str(binary),
        "--path",
        path,
        "--n",
        str(n),
        "--warmup",
        str(warmup),
        "--steps",
        str(steps),
        "--thermo-every",
        "1000",
    ]
    if dump:
        cmd.extend(["--dump", dump])
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    line = completed.stdout.strip().splitlines()[-1]
    return json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "build-m8" / "tilemd_bench.exe")
    parser.add_argument("--sizes", nargs="+", type=int, default=[32000, 256000, 2_048_000])
    parser.add_argument("--paths", nargs="+", default=list(PATHS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=0, help="override the section 8 step count")
    parser.add_argument("--dump-32k", type=Path, default=ROOT / "tests/validation/lammps_ref/init_32k.dump")
    args = parser.parse_args()
    out_dir = ROOT / "bench" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in args.sizes:
        steps = args.steps if args.steps > 0 else PROTOCOL[n]
        dump = str(args.dump_32k) if n == 32000 and args.dump_32k.exists() else ""
        for path in args.paths:
            runs = []
            for repeat in range(args.repeats):
                print(f"run {path} n={n} steps={steps} repeat={repeat + 1}", flush=True)
                try:
                    sample = run_once(args.binary, path, n, steps, args.warmup, dump)
                except (RuntimeError, json.JSONDecodeError, KeyError) as exc:
                    sample = {"path": path, "n": n, "steps": steps, "error": str(exc)}
                    print(sample["error"], flush=True)
                sample["repeat"] = repeat + 1
                runs.append(sample)
                if "error" in sample:
                    break
            rates = [row["atom_steps_per_s"] for row in runs if "atom_steps_per_s" in row]
            payload = {
                "code": "tilemd",
                "milestone": "speed-baseline",
                "system": {"name": "lj_melt", "N": n, "rho": 0.8442, "rc": 2.5, "skin": 0.3, "dt": 0.005},
                "settings": {
                    "path": path,
                    "list": "full",
                    "graph_unroll": 50,
                    "rebuild_every_step": False,
                    "warmup": args.warmup,
                    "protocol_steps": PROTOCOL.get(n, steps),
                    "note": "fused-tile is the cluster-tile Run. There is no separate tile integrator.",
                },
                "runs": runs,
                "median_atom_steps_per_s": median(rates) if rates else None,
            }
            dest = out_dir / f"2026-10-19_rtx3050_baseline_{path}_n{n}.json"
            dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {dest} median {payload['median_atom_steps_per_s']}", flush=True)
            rows.append(payload)
    write_speed(rows)
    return 0


def write_speed(rows: list[dict]) -> None:
    by_key = {}
    for path in (ROOT / "bench" / "results").glob("2026-10-19_rtx3050_baseline_*.json"):
        loaded = json.loads(path.read_text(encoding="utf-8"))
        by_key[(loaded["settings"]["path"], loaded["system"]["N"])] = loaded
    for row in rows:
        by_key[(row["settings"]["path"], row["system"]["N"])] = row
    lines = [
        "# Speed",
        "",
        "Baseline of the paths that exist on this tree. Timer is host wall time around the step loop after warmup. Clocks were not locked. GPUMD and LAMMPS-KOKKOS have not been run on this machine. `fused-tile` is the cluster-tile Run; there is no separate tile integrator.",
        "",
        "| path | GPU | N | steps | atom-steps/s | vs GPUMD | vs LAMMPS |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    rows = [by_key[key] for key in sorted(by_key, key=lambda item: (item[1], item[0]))]
    for row in rows:
        run = next((item for item in row["runs"] if "steps" in item), {})
        rate = row["median_atom_steps_per_s"]
        rate_text = f"{rate:.6e}" if isinstance(rate, float) else "not measured"
        lines.append(
            f"| {row['settings']['path']} | RTX 3050 | {row['system']['N']} | {run.get('steps', '')} | {rate_text} | — | — |"
        )
    lines.append("")
    lines.append("Bytes per atom-step and instructions per atom-step wait on a Nsight Compute capture of the FP32 path.")
    lines.append("")
    dest = ROOT / "reports" / "SPEED.md"
    dest.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
