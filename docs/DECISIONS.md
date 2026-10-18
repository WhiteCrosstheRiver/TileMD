# Decision log

Fixed decisions live in [PLAN.md](PLAN.md) section 2. This file records changes to those decisions. Each change needs the measurement that justified it.

| ID | Date | Change | Evidence | Supersedes |
|---|---|---|---|---|
| D3 | 2026-10-19 | EAM uses LAMMPS metal units (eV, Å). Lennard-Jones stays in reduced units. | `Cu_u3.eam` forces match LAMMPS `pair_style eam` to 1.4e-13 relative on the 108-atom cell in `reports/M8.md`. | D3's postponement of metal units, for the EAM comparison only. |
