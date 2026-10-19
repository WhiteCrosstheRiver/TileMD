"""Copper EAM NVE drift inputs and slope comparison.

The 108-atom cell and the velocities match the 100-step LAMMPS reference.
Thermo samples are total energy per atom. The slope is a least-squares fit
of that energy against time.
"""

import argparse
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
START = ROOT / "tests" / "validation" / "lammps_ref" / "cu_u3_108.txt"
POTENTIAL = ROOT / "tests" / "potentials" / "Cu_u3.eam"


def load_start():
    values = START.read_text().split()
    n = int(values[0])
    box = (float(values[1]), float(values[2]), float(values[3]))
    cursor = 5
    x = []
    y = []
    z = []
    for _ in range(n):
        x.append(float(values[cursor]))
        y.append(float(values[cursor + 1]))
        z.append(float(values[cursor + 2]))
        cursor += 6
    return n, box, x, y, z


def copper_velocity(n):
    vx = [0.8 * ((i % 7) - 3) for i in range(n)]
    vy = [0.5 * ((i % 5) - 2) for i in range(n)]
    vz = [0.4 * ((i % 3) - 1) for i in range(n)]
    mx = sum(vx) / n
    my = sum(vy) / n
    mz = sum(vz) / n
    return [v - mx for v in vx], [v - my for v in vy], [v - mz for v in vz]


def write_inputs(out_dir, steps, thermo_every):
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    n, box, x, y, z = load_start()
    vx, vy, vz = copper_velocity(n)
    xyz = out_dir / "cu108.extxyz"
    with xyz.open("w", encoding="ascii") as out:
        out.write(f"{n}\n")
        out.write(
            f'Lattice="{box[0]:.17e} 0 0 0 {box[1]:.17e} 0 0 0 {box[2]:.17e}" '
            "Properties=species:S:1:pos:R:3:vel:R:3:type:I:1 shift=none pbc=\"T T T\"\n"
        )
        for i in range(n):
            out.write(
                f"A {x[i]:.17e} {y[i]:.17e} {z[i]:.17e} "
                f"{vx[i]:.17e} {vy[i]:.17e} {vz[i]:.17e} 0\n"
            )
    tilemd_in = out_dir / "in.cu_drift"
    tilemd_in.write_text(
        "\n".join(
            [
                "units metal",
                f"structure {xyz}",
                f"eam {POTENTIAL}",
                "timestep 0.001",
                f"thermo_every {thermo_every}",
                f"run {steps}",
                "",
            ]
        ),
        encoding="ascii",
    )
    data = out_dir / "cu108.data"
    with data.open("w", encoding="ascii") as out:
        out.write("Cu 3x3x3, one atom displaced\n\n")
        out.write(f"{n} atoms\n1 atom types\n\n")
        out.write(f"0 {box[0]:.17e} xlo xhi\n")
        out.write(f"0 {box[1]:.17e} ylo yhi\n")
        out.write(f"0 {box[2]:.17e} zlo zhi\n\n")
        out.write("Masses\n\n1 63.55\n\n")
        out.write("Atoms # atomic\n\n")
        for i in range(n):
            out.write(f"{i + 1} 1 {x[i]:.17e} {y[i]:.17e} {z[i]:.17e}\n")
        out.write("\nVelocities\n\n")
        for i in range(n):
            out.write(f"{i + 1} {vx[i]:.17e} {vy[i]:.17e} {vz[i]:.17e}\n")
    lammps_in = out_dir / "in.cu_eam_drift"
    lammps_in.write_text(
        "\n".join(
            [
                "units metal",
                "atom_style atomic",
                "boundary p p p",
                f"read_data {data}",
                "pair_style eam",
                f"pair_coeff * * {POTENTIAL}",
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
    return tilemd, lammps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--thermo-every", type=int, default=100)
    parser.add_argument("--compare", nargs=2, metavar=("TILEMD", "LAMMPS"))
    parser.add_argument("--write", type=pathlib.Path)
    args = parser.parse_args()
    if args.compare:
        compare(pathlib.Path(args.compare[0]), pathlib.Path(args.compare[1]), args.write)
        return
    tilemd_in, lammps_in = write_inputs(args.out_dir, args.steps, args.thermo_every)
    print(tilemd_in)
    print(lammps_in)


if __name__ == "__main__":
    main()
