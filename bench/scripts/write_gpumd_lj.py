"""Write GPUMD inputs for the LJ melt in argon metal units.

The numbers follow PLAN.md section 4.4. This script has not been run
through a GPUMD binary. The cloned commit must be recorded when T-B2
builds GPUMD, and the skin line must be checked against that commit:
GPUMD v4.9 hard-codes a 1 Angstrom skin for NEP, and the historical
``neighbor <skin>`` keyword may be absent.

32k uses tests/validation/lammps_ref/init_32k.dump, the same frame the
TileMD bench times. Other sizes use the same fcc as bench/tilemd_bench.cu
and let GPUMD draw velocities at the temperature of that bench
(unit-Gaussian reduced velocities, T* = 1), not the PLAN T* = 1.44 melt.
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DUMP_32K = ROOT / "tests" / "validation" / "lammps_ref" / "init_32k.dump"
POTENTIAL = ROOT / "bench" / "inputs" / "gpumd" / "lj.txt"

EPSILON_EV = 0.010323
SIGMA_A = 3.405
MASS_AMU = 39.948
# LAMMPS metal: kinetic energy (eV) = 0.5 * m(amu) * v(A/ps)^2 * MVV2E.
MVV2E = 1.0364269e-4
KB_EV_PER_K = 8.617333262145e-5
RHO = 0.8442
RC_REDUCED = 2.5
SKIN_REDUCED = 0.3
DT_REDUCED = 0.005
PROTOCOL = {32_000: 100_000, 256_000: 20_000, 2_048_000: 5_000}
WARMUP = 1000


def tau_ps() -> float:
    return SIGMA_A * math.sqrt(MASS_AMU * MVV2E / EPSILON_EV)


def dt_fs() -> float:
    return DT_REDUCED * tau_ps() * 1000.0


def temperature_k(t_reduced: float) -> float:
    return t_reduced * EPSILON_EV / KB_EV_PER_K


def velocity_scale_a_per_fs() -> float:
    return SIGMA_A / (tau_ps() * 1000.0)


def box_length(n: int) -> float:
    return (n / RHO) ** (1.0 / 3.0)


def read_dump(path: Path) -> tuple[float, list[tuple[float, float, float, float, float, float]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines[2] != "ITEM: NUMBER OF ATOMS":
        raise RuntimeError(f"unexpected dump header: {path}")
    n = int(lines[3])
    box = float(lines[5].split()[1])
    atoms = []
    for line in lines[9:]:
        parts = line.split()
        if len(parts) < 8:
            continue
        atoms.append(tuple(float(parts[i]) for i in range(2, 8)))
    if len(atoms) != n:
        raise RuntimeError(f"dump has {len(atoms)} atoms, header says {n}")
    return box, atoms


def fcc_positions(n: int) -> tuple[float, list[tuple[float, float, float]]]:
    cells = round((n / 4.0) ** (1.0 / 3.0))
    if cells * cells * cells * 4 != n:
        raise RuntimeError(f"{n} is not 4 * cells^3")
    length = box_length(n)
    spacing = length / cells
    basis = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5))
    atoms = []
    for ix in range(cells):
        for iy in range(cells):
            for iz in range(cells):
                for bx, by, bz in basis:
                    atoms.append(((ix + bx) * spacing, (iy + by) * spacing, (iz + bz) * spacing))
    return length, atoms


def write_xyz(path: Path, length_reduced: float, rows: list[str], with_velocity: bool) -> None:
    length = length_reduced * SIGMA_A
    props = "species:S:1:pos:R:3:mass:R:1:vel:R:3" if with_velocity else "species:S:1:pos:R:3:mass:R:1"
    header = f'Lattice="{length} 0 0 0 {length} 0 0 0 {length}" Properties={props} pbc="T T T"'
    path.write_text(f"{len(rows)}\n{header}\n" + "\n".join(rows) + "\n", encoding="utf-8")


def write_run(path: Path, steps: int) -> None:
    skin = SKIN_REDUCED * SIGMA_A
    text = f"""# Argon LJ melt. Not yet executed. tau = {tau_ps():.6f} ps, dt = {dt_fs():.6f} fs.
# Cutoff {RC_REDUCED}*sigma = {RC_REDUCED * SIGMA_A:.4f} A. Requested skin {skin:.4f} A.
# If this GPUMD commit rejects `neighbor`, delete that line and record the built-in skin.
potential lj.txt
time_step {dt_fs():.6f}

ensemble nve
dump_thermo {WARMUP}
neighbor {skin:.4f}
run {WARMUP}

ensemble nve
dump_thermo {WARMUP}
neighbor {skin:.4f}
run {steps}
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, action="append", dest="sizes")
    parser.add_argument("--out", type=Path, default=ROOT / "bench" / "inputs" / "gpumd" / "generated")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sizes = args.sizes or [32_000, 256_000, 2_048_000]
    print(
        f"tau_ps {tau_ps():.6f} dt_fs {dt_fs():.6f} "
        f"T_K_at_Tstar_1 {temperature_k(1.0):.4f} "
        f"T_K_at_Tstar_1.44 {temperature_k(1.44):.4f} "
        f"v_scale_A_per_fs {velocity_scale_a_per_fs():.8e}"
    )
    if DUMP_32K.is_file():
        box = float(DUMP_32K.read_text(encoding="utf-8").splitlines()[5].split()[1])
        print(f"dump_32k_box {box:.16e} generated_box {box_length(32_000):.16e}")
    if not args.write:
        print("check only; pass --write to emit model.xyz and run.in")
        return 0
    scale = velocity_scale_a_per_fs()
    for n in sizes:
        dest = args.out / f"n{n}"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(POTENTIAL, dest / "lj.txt")
        steps = PROTOCOL[n]
        if n == 32_000 and DUMP_32K.is_file():
            length, atoms = read_dump(DUMP_32K)
            rows = [
                f"Ar {x * SIGMA_A:.9f} {y * SIGMA_A:.9f} {z * SIGMA_A:.9f} "
                f"{MASS_AMU:.6f} {vx * scale:.9e} {vy * scale:.9e} {vz * scale:.9e}"
                for x, y, z, vx, vy, vz in atoms
            ]
            write_run(dest / "run.in", steps)
            write_xyz(dest / "model.xyz", length, rows, with_velocity=True)
        else:
            length, atoms = fcc_positions(n)
            rows = [
                f"Ar {x * SIGMA_A:.9f} {y * SIGMA_A:.9f} {z * SIGMA_A:.9f} {MASS_AMU:.6f}"
                for x, y, z in atoms
            ]
            # C++ mt19937 velocities are not reproduced here. GPUMD draws T* = 1.
            write_run(dest / "run.in", steps)
            run = (dest / "run.in").read_text(encoding="utf-8")
            run = run.replace(
                "potential lj.txt\n",
                f"potential lj.txt\nvelocity {temperature_k(1.0):.6f} seed 20261018\n",
            )
            (dest / "run.in").write_text(run, encoding="utf-8")
            write_xyz(dest / "model.xyz", length, rows, with_velocity=False)
        print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
