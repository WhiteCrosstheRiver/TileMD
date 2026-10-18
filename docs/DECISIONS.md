# Decision log

Fixed decisions live in [PLAN.md](PLAN.md) section 2. This file records changes to those decisions. Each change needs the measurement that justified it.

| ID | Date | Change | Evidence | Supersedes |
|---|---|---|---|---|
| D3 | 2026-10-19 | EAM uses LAMMPS metal units (eV, Å). Lennard-Jones stays in reduced units. | `Cu_u3.eam` forces match LAMMPS `pair_style eam` to 1.4e-13 relative on the 108-atom cell in `reports/M8.md`. | D3's postponement of metal units, for the EAM comparison only. |
| D17 | 2026-10-19 | `pos4.w` stores the shadow cell, not the type. xyz is the offset from a cell of edge 4. | On `melt_4000` shifted by 256, absolute binary32 force error is `1.64e-3`. The cell-local shadow is `9.14e-6`, inside T1. See `reports/M6.md`. | The type-as-int-bits field in D4 and the §5.1 note that w holds type bits. The type array is unchanged. |
