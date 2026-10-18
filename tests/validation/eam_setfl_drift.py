"""Cu-Ni setfl NVE drift inputs and slope comparison.

Four atoms, the same start as the 100-step LAMMPS reference. Thermo samples
are total energy per atom. The slope is a least-squares fit against time.
"""

import argparse
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
INSTALLED = pathlib.Path(
    "C:/ProgramData/anaconda3/Lib/site-packages/lammps/share/lammps/potentials/CuNi.eam.alloy"
)
BUNDLED = ROOT / "tests" / "potentials" / "CuNi.eam.alloy"
LAMMPS = pathlib.Path(r"C:\ProgramData\anaconda3\Lib\site-packages\lammps\lmp.exe")
TILEMD = ROOT / "build-m8" / "tilemd.exe"

X = [10.0, 12.2, 10.4, 13.1]
Y = [10.0, 10.1, 12.5, 12.0]
Z = [10.0, 10.3, 10.2, 11.4]
TYPE = [0, 1, 0, 1]
VX = [-1.2158939747208246, -0.41589397472082434, 0.3841060252791757, 1.1841060252791757]
VY = [-0.75993373420051535, -0.25993373420051535, 0.24006626579948462, 0.74006626579948465]
VZ = [-0.29602650631979388, 0.10397349368020617, 0.50397349368020616, -0.29602650631979388]


def potential():
    if BUNDLED.exists():
        return BUNDLED
    if INSTALLED.exists():
        return INSTALLED
    raise SystemExit("CuNi.eam.alloy is not available")


def write_inputs(out_dir, steps, thermo_every):
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    table = potential()
    xyz = out_dir / "cuni4.extxyz"
    with xyz.open("w", encoding="ascii") as out:
        out.write("4\n")
        out.write(
            'Lattice="20 0 0 0 20 0 0 0 20" '
            'Properties=species:S:1:pos:R:3:vel:R:3:type:I:1 shift=none pbc="T T T"\n'
        )
        for i in range(4):
            species = "A" if TYPE[i] == 0 else "B"
            out.write(
                f"{species} {X[i]:.17e} {Y[i]:.17e} {Z[i]:.17e} "
                f"{VX[i]:.17e} {VY[i]:.17e} {VZ[i]:.17e} {TYPE[i]}\n"
            )
    tilemd_in = out_dir / "in.cuni_drift"
    tilemd_in.write_text(
        "\n".join(
            [
                "units metal",
                f"structure {xyz}",
                f"eam {table}",
                "timestep 0.001",
                f"thermo_every {thermo_every}",
                f"run {steps}",
                "",
            ]
        ),
        encoding="ascii",
    )
    data = out_dir / "cuni4.data"
    with data.open("w", encoding="ascii") as out:
        out.write("CuNi 4-atom NVE\n\n")
        out.write("4 atoms\n2 atom types\n\n")
        out.write("0.0 20.0 xlo xhi\n0.0 20.0 ylo yhi\n0.0 20.0 zlo zhi\n\n")
        out.write("Masses\n\n1 58.689\n2 63.546\n\n")
        out.write("Atoms # atomic\n\n")
        for i in range(4):
            out.write(f"{i + 1} {TYPE[i] + 1} {X[i]:.17e} {Y[i]:.17e} {Z[i]:.17e}\n")
        out.write("\nVelocities\n\n")
        for i in range(4):
            out.write(f"{i + 1} {VX[i]:.17e} {VY[i]:.17e} {VZ[i]:.17e}\n")
    lammps_in = out_dir / "in.cuni_eam_drift"
    lammps_in.write_text(
        "\n".join(
            [
                "units metal",
                "atom_style atomic",
                "boundary p p p",
                f"read_data {data}",
                "pair_style eam/alloy",
                f"pair_coeff * * {table.as_posix()} Ni Cu",
                "neighbor 0.3 bin",
                "neigh_modify delay 0 every 1 check yes",
                "timestep 0.001",
                "fix 1 all nve",
                f"thermo {thermo_every}",
                "thermo_style custom step time pe ke etotal",
                "thermo_modify norm yes format float %.16g",
                f"run {steps}",
                "",
            ]
        ),
        encoding="ascii",
    )
    return tilemd_in, lammps_in


def slope(xs, ys):
    count = len(xs)
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    numer = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom = sum((x - mean_x) ** 2 for x in xs)
    return numer / denom


def read_text(path):
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8")


def read_tilemd(path):
    rows = []
    for line in read_text(path).splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split()
        rows.append((int(parts[0]), float(parts[1]), float(parts[5])))
    return rows


def read_lammps(path):
    rows = []
    past_header = False
    for line in read_text(path).splitlines():
        line = line.strip()
        if line.startswith("Step "):
            past_header = True
            continue
        if not past_header:
            continue
        parts = line.split()
        if len(parts) < 5 or not parts[0].isdigit():
            if rows:
                break
            continue
        rows.append((int(parts[0]), float(parts[1]), float(parts[4])))
    return rows


def compare(tilemd_path, lammps_path, table_path=None):
    tilemd = read_tilemd(tilemd_path)
    lammps = read_lammps(lammps_path)
    if len(tilemd) != len(lammps):
        raise SystemExit(f"sample count {len(tilemd)} vs {len(lammps)}")
    times = [row[1] for row in tilemd]
    tilemd_e = [row[2] for row in tilemd]
    lammps_e = [row[2] for row in lammps]
    tilemd_slope = slope(times, tilemd_e)
    lammps_slope = slope(times, lammps_e)
    tilemd_span = max(abs(value - tilemd_e[0]) for value in tilemd_e)
    lammps_span = max(abs(value - lammps_e[0]) for value in lammps_e)
    max_gap = max(abs(a - b) for a, b in zip(tilemd_e, lammps_e))
    print(f"samples {len(tilemd)}")
    print(f"tilemd e0 {tilemd_e[0]:.16e} e_final {tilemd_e[-1]:.16e}")
    print(f"lammps e0 {lammps_e[0]:.16e} e_final {lammps_e[-1]:.16e}")
    print(f"tilemd slope {tilemd_slope:.16e} eV/atom/ps span {tilemd_span:.16e}")
    print(f"lammps slope {lammps_slope:.16e} eV/atom/ps span {lammps_span:.16e}")
    print(f"max |e_tilemd - e_lammps| {max_gap:.16e}")
    if table_path is not None:
        with table_path.open("w", encoding="ascii", newline="\n") as out:
            out.write("step time e_tilemd e_lammps\n")
            for left, right in zip(tilemd, lammps):
                out.write(f"{left[0]} {left[1]:.16e} {left[2]:.16e} {right[2]:.16e}\n")
    return tilemd_slope, lammps_slope, max_gap, tilemd_span


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--thermo-every", type=int, default=100)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--compare", nargs=2, metavar=("TILEMD", "LAMMPS"))
    parser.add_argument("--write", type=pathlib.Path)
    args = parser.parse_args()
    if args.compare:
        compare(pathlib.Path(args.compare[0]), pathlib.Path(args.compare[1]), args.write)
        return
    tilemd_in, lammps_in = write_inputs(args.out_dir, args.steps, args.thermo_every)
    print(tilemd_in)
    print(lammps_in)
    if not args.run:
        return
    out_dir = args.out_dir.resolve()
    tilemd_out = out_dir / "tilemd.txt"
    lammps_log = out_dir / "lammps.log"
    with tilemd_out.open("w", encoding="utf-8") as thermo:
        subprocess.run([str(TILEMD), str(tilemd_in)], check=True, stdout=thermo)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    subprocess.run([str(LAMMPS), "-in", str(lammps_in), "-log", str(lammps_log)], check=True, env=env)
    compare(tilemd_out, lammps_log, args.write)


if __name__ == "__main__":
    main()
