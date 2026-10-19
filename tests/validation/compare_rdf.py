#!/usr/bin/env python3
"""L1 distance between TileMD and LAMMPS g(r) on [0, rmax]."""

from __future__ import annotations

import argparse
import sys


def load_tilemd(path: str) -> list[tuple[float, float]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            r_mid, g_r, _coord = line.split()[:3]
            rows.append((float(r_mid), float(g_r)))
    return rows


def load_lammps(path: str) -> list[tuple[float, float]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            # row index, r, g(r), coordination
            rows.append((float(parts[1]), float(parts[2])))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tilemd")
    parser.add_argument("lammps")
    args = parser.parse_args()
    got = load_tilemd(args.tilemd)
    ref = load_lammps(args.lammps)
    if len(got) != len(ref):
        print(f"bin count {len(got)} != {len(ref)}", file=sys.stderr)
        return 1
    acc = 0.0
    max_abs = 0.0
    for (r_got, g_got), (r_ref, g_ref) in zip(got, ref):
        if abs(r_got - r_ref) > 1e-6:
            print(f"bin center {r_got} != {r_ref}", file=sys.stderr)
            return 1
        delta = abs(g_got - g_ref)
        max_abs = max(max_abs, delta)
        acc += delta
    if len(got) < 2:
        print("no bins", file=sys.stderr)
        return 1
    dr = got[1][0] - got[0][0]
    l1 = acc * dr
    print(f"bins {len(got)} dr {dr:.6g} max_abs {max_abs:.6g} L1 {l1:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
