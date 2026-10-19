Superseded in part by docs/REVIEW_AND_PLAN_v2.md

# TileMD — Implementation Plan (Stage 1: LJ validation engine)

> **Audience:** the coding agent(s) implementing this engine, and the reviewer (Claude) who will review every pull request on GitHub.
> **Location in repo:** commit this file as `docs/PLAN.md` in the first commit. Keep the **Status** table below up to date in every PR.
> **Working name:** `TileMD` (rename freely; if renamed, rename everywhere in one commit).

---

## 0. Status (update in every PR)

| Milestone | Branch | State | Gate | PR | Notes |
|---|---|---|---|---|---|
| M0 Repo bootstrap | `m0-bootstrap` | in progress | — | | Local check passed. The input parser runs the CPU reference. No pull request yet. |
| M1 Baselines (GPUMD, LAMMPS-KOKKOS) | `m1-baselines` | not started | — | | |
| M2 CPU reference oracle | `m2-cpu-reference` | in progress | G1-ref | | CPU oracle matches LAMMPS. Gate G1-ref passed locally. No pull request yet. |
| M3 GPU v0 (correct, simple) | `m3-gpu-v0` | in progress | G1 | | Double-path drift passes T4-v2, so G1 is passed. The old slope ratio was 2.33. |
| M4 Performance foundation | `m4-perf-foundation` | in progress | G2, G3 | | Graphs, device rebuild, and block 128 are in. The 2M run and Nsight Systems remain open. |
| M5 Cluster-pair tiles (SIMT) | `m5-cluster-tiles` | in progress | G2+ | | Tiles match the oracle. Efficiency is about 0.09, and the tile run is slower. |
| M6 Fused single-kernel timestep | `m6-fused-step` | in progress | G2+, G3+ | | Fused leapfrog matches a small CPU run. On this 3050 it is slower than the atom path. |
| M7 Tensor-core pair-attention tile | `m7-tensor-core` | in progress | G4 | | CPU 3×TF32 misses T1 and is off the default path. No tensor-core kernel. |
| M8 Generic pair-attention API + EAM | `m8-pair-attention-api` | in progress | G5 | | EAM matches LAMMPS. Device kernels launched. The G5 overhead gate was missed. |

States: `not started` → `in progress` → `in review` → `changes requested` → `merged`.

---

## 1. Purpose, scope, non-goals

### 1.1 Purpose
Build a **new, general GPU molecular-dynamics engine from scratch** whose force evaluation is organized as dense, tiled, register/shared-memory-resident computation ("pair-attention"), so that:
1. simple potentials (LJ) run near the GPU memory/compute roofline, far faster than GPUMD's LJ;
2. later, many-body potentials (EAM, then NEP with its functional form unchanged) reuse the same tiled primitive, and dense parts are mapped onto tensor cores.

### 1.2 Why LJ first
LJ has ~25 FLOPs per pair, so any slowness is engine overhead, memory traffic or synchronization — not physics. It isolates engine quality. Evidence for headroom: GPUMD's LJ is only ~10× faster than its NEP although NEP needs ~40–100× more arithmetic per atom, so GPUMD's LJ is bound by overhead/memory, not math. Known GPUMD LJ costs (from `src/force/lj.cu`, `src/force/neighbor.cu`):
- positions gathered as `double`;
- 9-component per-atom virial + per-atom energy written (double read-modify-write) every step;
- the neighbor-rebuild check copies a value to the host every step (host sync each step);
- several separate kernels per step (force, check, integrator stages);
- no spatial sorting of atoms; one thread per atom.

### 1.3 Stage-1 scope (must have)
- LJ (single and multi-type), orthorhombic periodic box, reduced LJ units.
- NVE (velocity Verlet / leapfrog) and Langevin NVT.
- Thermo output (T, PE, KE, total E, pressure), extended-XYZ dumps.
- Correctness validated against a CPU reference and LAMMPS.
- Benchmarks against GPUMD and LAMMPS-KOKKOS on the same GPU.

### 1.4 Non-goals for stage 1 (do NOT implement yet)
- Triclinic boxes, long-range electrostatics, constraints, NPT barostat, multi-GPU, NEP, Tersoff, metal units (reserved for later stages; keep code structure open for them).

---

## 2. Decisions already made (decision log)

The reviewer holds decision authority for this project. These are fixed unless a PR provides measured evidence to change them; record any change in `docs/DECISIONS.md` with the evidence.

| ID | Decision | Rationale |
|---|---|---|
| D1 | Language: CUDA C++17, CUDA ≥ 12.4, CMake ≥ 3.24, target archs `sm_80;sm_90`. Python 3.11 for scripts/analysis only. | H100/A100 targets; CUDA 12.4 gives conditional graph nodes. |
| D2 | Use CUB for scans/sorts/reductions. No Thrust in hot paths. No other runtime dependencies. | Fast, header-only, ships with CUDA. |
| D3 | Units: reduced LJ units (ε = σ = m = k_B = 1) in stage 1. | Unit-independent comparison; metal units later. |
| D4 | Mixed precision: positions & velocities stored as `double` (master copy); forces computed from a `float4` shadow `(x, y, z, type-as-int-bits)` of wrapped positions; per-thread force accumulation in `float`; global reductions (energy, virial, KE) in `double`. | Standard mixed precision (HOOMD/GROMACS style): FP32 speed, FP64 long-run stability. |
| D5 | Full neighbor lists (each pair evaluated from both sides, no atomics on forces) in M3–M6. Half lists only as a measured experiment. | LAMMPS-KOKKOS finds full list + Newton off usually best on GPUs; gives bitwise determinism. |
| D6 | Integrator: velocity Verlet in M3–M5; leapfrog (trajectory-equivalent) in M6 because it fuses into one kernel. | See M6. |
| D7 | Neighbor skin default 0.3σ; rebuild when max displacement since last build > skin/2. Decision computed on device. | Standard; removes per-step host sync. |
| D8 | **Determinism required:** same input + seed + GPU model + binary ⇒ bitwise-identical trajectory. No floating-point atomics in any per-step path. | Debuggability and reproducibility for publication. |
| D9 | Atoms are internally reordered (spatial sort); all output is written in original global-ID order. | Sorting must be invisible to users. |
| D10 | Orthorhombic box only; minimum image via `d -= L * rintf(d * Linv)`. | Stage-1 scope. |
| D11 | Cluster size 8, GROMACS-style column decomposition (M5). | 8×8 tiles map to warps and tensor-core fragments. |
| D12 | Tensor-core path uses 3×TF32 split precision and is **optional**: it is kept only if gate G4 passes. | Plain TF32 (10-bit mantissa) is too coarse for LJ force cancellation. |
| D13 | Thermostats in stage 1: NVE and Langevin only (Langevin is atom-local, fuses cleanly). Nosé–Hoover later with a proper two-pass scheme. | Avoid lagged global-reduction thermostats. |
| D14 | RNG: counter-based Philox (`curand_philox4x32_10` or hand-rolled), keyed by `(seed, step, global_id)`. | Deterministic, order-independent random numbers. |
| D15 | I/O: simple key-value input file (`*.in`), extended XYZ for structures and dumps, CSV for thermo, JSON for benchmark results. | Readable by ASE/OVITO/pandas. |
| D16 | Primary benchmark GPU: H100 SXM; secondary: A100. | User's cluster hardware. |

---

## 3. Repository layout (create in M0)

```
TileMD/
├── CMakeLists.txt
├── README.md                    # build, run, test, benchmark in <10 commands
├── CHANGELOG.md
├── LICENSE                      # choose with the user (default GPL-3.0 to match GPUMD lineage, or MIT)
├── .clang-format                # Google style, ColumnLimit 100
├── .gitignore                   # see Appendix A
├── .github/
│   ├── pull_request_template.md # see Appendix B
│   ├── ISSUE_TEMPLATE/decision.md
│   └── workflows/ci.yml         # build check + CPU tests (no GPU on hosted runners)
├── docs/
│   ├── PLAN.md                  # this file
│   ├── DECISIONS.md             # changes to the decision log, with evidence
│   ├── DESIGN.md                # living design doc: data layout, kernels, invariants
│   └── NUMERICS.md              # precision analysis, error budgets (M7)
├── src/
│   ├── core/        # Box, SystemState (device arrays), Params, error/status word
│   ├── io/          # input parser, extxyz reader/writer, thermo writer
│   ├── neighbor/    # cell grid, spatial sort, neighbor list (M3), cluster-pair list (M5)
│   ├── force/       # lj_naive.cu (M3), lj_tile.cu (M5), lj_tc.cu (M7), pair_attention/ (M8), eam/ (M8)
│   ├── integrate/   # verlet.cu, langevin.cu, fused_step.cu (M6)
│   ├── run/         # Simulation driver, graph capture, segment loop
│   └── main.cu
├── reference/       # M2 CPU oracle (C++ or NumPy), deliberately simple
├── tests/
│   ├── unit/        # GoogleTest (fetched by CMake) — CPU + GPU tests
│   ├── fixtures/    # small extxyz configs + reference forces/energies (≤ 1 MB each)
│   └── validation/  # scripts for NVE drift, RDF, pressure vs LAMMPS
├── bench/
│   ├── inputs/      # TileMD, GPUMD and LAMMPS inputs for identical systems
│   ├── scripts/     # run_bench.py, collect_env.py, plot_bench.py
│   └── results/     # JSON results (committed), see §8
├── reports/         # one markdown report per milestone (committed)
└── scripts/
    ├── check.sh     # format check + build + CPU tests (runs in CI)
    └── gpu_check.sh # full GPU test suite; output log pasted into PR
```

---

## 4. Physics specification (exact formulas — implement exactly this)

### 4.1 LJ pair potential (reduced units)
For a pair of types (a, b) with parameters ε_ab, σ_ab, cutoff r_c,ab:

- `U(r) = 4 ε [ (σ/r)^12 − (σ/r)^6 ] − U_shift` for r < r_c, else 0.
- Energy-shift modes (input key `lj_shift`):
  - `none` (default, identical to LAMMPS `lj/cut` with `pair_modify shift no`): U_shift = 0.
  - `energy`: U_shift = 4ε[(σ/r_c)^12 − (σ/r_c)^6]. Forces unchanged.
  - `force` (shifted-force, used for drift tests): U_sf(r) = U(r) − U(r_c) − (r − r_c)·U'(r_c); force F_sf(r) = F(r) − F(r_c).
- Force on atom i from j: **F_ij = s(r²) · r_ij**, with r_ij = r_i − r_j (minimum image) and
  `s(r²) = 24 ε [ 2 σ^12 / r^14 − σ^6 / r^8 ]` (multiply by r_ij, not by unit vector).
- Implement via `inv_r2 = 1/r²`, `sr6 = (σ² · inv_r2)^3`, `s = 24 ε · inv_r2 · sr6 · (2·sr6 − 1)`, `u = 4 ε · sr6 · (sr6 − 1)`.
- Precompute per type pair: `sigma2`, `eps24`, `eps4`, `rc2`, shift constants; store in `__constant__` memory (max 8 types in stage 1).
- Mixing: explicit per-pair table in input; no implicit mixing rules.

### 4.2 Observables
- Per-atom energy (full list): `e_i = ½ Σ_j u_ij`; total PE = Σ e_i.
- Virial tensor (global): `W_ab = ½ Σ_i Σ_j (r_ij)_a (F_ij)_b` (full list ⇒ ½).
- Pressure: `P = (N k_B T + (W_xx+W_yy+W_zz)/3) / V`, with `T = 2 KE / (k_B · (3N − 3))` (momentum conserved).
- Per-atom virial only on dump steps that request it (template flag).

### 4.3 Integrators
- **Velocity Verlet (M3–M5):**
  1. `v += ½ dt · f/m`; 2. `x += dt · v`; wrap into box, update image counters; refresh float4 shadow; 3. compute f(x); 4. `v += ½ dt · f/m`.
- **Leapfrog (M6):** state holds x(t), v(t − ½dt). Per step: compute f(t) from x(t); `v(t+½) = v(t−½) + dt·f/m`; `x(t+dt) = x(t) + dt·v(t+½)`. Kinetic energy at integer time for thermo: `v(t) = ½ (v(t−½) + v(t+½))`.
  - Conversion at start: `v(−½) = v(0) − ½ dt · f(0)/m` (requires one initial force call).
- **Langevin (BAOAB splitting preferred; document choice in DESIGN.md):** friction γ, target T; random force from Philox keyed by (seed, step, global_id) (D14).

### 4.4 Standard benchmark system: LJ melt (identical to LAMMPS `bench/in.lj`)
- fcc lattice, reduced density ρ* = 0.8442; N = 4·n³ with n ∈ {20, 40, 80, 160} → N ∈ {32 000, 256 000, 2 048 000, 16 384 000}.
- Initial velocities: Gaussian at T* = 1.44 (seeded), zero total momentum.
- r_c = 2.5σ, skin = 0.3σ, dt = 0.005τ, NVE, `lj_shift = none`.
- For GPUMD (metal units): argon mapping ε = 0.010323 eV, σ = 3.405 Å, m = 39.948 amu; convert density, dt, cutoff accordingly; GPUMD default skin 1.0 Å ≈ 0.29σ — set explicitly to 0.3σ = 1.0215 Å if configurable, otherwise record what was used.

---

## 5. Data layout specification

### 5.1 Device state (`SystemState`)
| Array | Type | Size | Notes |
|---|---|---|---|
| `x` | `double3` (SoA: 3 × double[N] recommended) | N | master positions, wrapped |
| `v` | `double3` SoA | N | velocities |
| `img` | `int3` | N | image counters for unwrapped coordinates |
| `pos4` | `float4` | N | shadow: xyz is the offset from a cell of edge 4, w is the packed cell (`__int_as_float`). The type stays in the type array. |
| `f` | `float3` SoA (M3–M5); removed in M6 | N | forces |
| `gid` | `int` | N | original global ID of the atom at this sorted slot |
| `slot_of_gid` | `int` | N | inverse permutation (for output) |
| `x_ref` | `float4` | N | positions at last neighbor build (displacement check) |
| `status` | struct in device memory | 1 | flags: `need_rebuild`, `overflow`, `nan_detected`, counters |
| `thermo_ring` | struct array | R records | per-output-step reductions (double), drained by host at segment end |

### 5.2 Status word (single device struct, read by host once per segment)
```
struct DeviceStatus {
  int   need_rebuild;      // set by displacement check, cleared by rebuild
  int   overflow;          // neighbor or cluster capacity exceeded
  int   nan_detected;      // any non-finite force/position
  int   rebuild_count;     // for statistics
  float max_disp2_bits;    // as int bits for atomicMax on non-negative floats
};
```
- **Rule:** no `cudaMemcpy` device→host inside the time-step loop. Only at segment boundaries (§6.2).

### 5.3 Neighbor list (M3/M4)
- Layout: column-major `nl[k * N + i]` (k-th neighbor of atom i) for coalesced reads; `nn[i]` = count.
- Capacity `max_neigh` from input or auto = ceil(1.5 × (4/3)π(r_c+skin)³ ρ) rounded up to multiple of 32.
- On overflow: set `status.overflow`, keep running (results of the segment are discarded, see §6.2).

### 5.4 Cluster-pair structure (M5, GROMACS-style)
- Grid of columns in x–y; column side ≈ chosen so a column holds ~ 8·k atoms; within a column, atoms sorted by z; cut into clusters of 8 consecutive atoms; pad each column to a multiple of 8 with **dummy atoms** (type = `MASK_TYPE`, position far away / masked) so clusters never cross columns.
- Per cluster: bounding box (`float4 lo, hi`).
- Pair list: CSR — `ci_start[ci]`, `ci_end[ci]` into array `cj[]` of j-cluster indices plus `shift[]` index (0..26) for periodic image of the j-cluster (shift vectors precomputed per box).
- Interaction masks: 64-bit per (ci, cj) pair only needed for self-cluster (exclude i == j) and padding; padding can instead be masked by type. Document chosen approach.
- Pruning: include cj only if bbox–bbox distance < r_c + skin.

---

## 6. Host-side run loop (natural-language pseudocode)

### 6.1 Setup
```
read input file → Params
read or generate structure (extxyz or fcc lattice) → host arrays (double)
assign global IDs 0..N−1
copy to device; build pos4 shadow
build neighbor structure (cells → spatial sort/permutation → list)   [records x_ref]
compute initial forces; if leapfrog: convert v(0) → v(−½)
capture the step graph:
    stream-capture ONE time step (all kernels of that step, in order)
    instantiate a graph that contains K copies of the step (K = graph_unroll, default 50)
write thermo record for step 0
```

### 6.2 Segment loop
```
segment_length S = min(steps until next host event, segment_max)  (default segment_max = 1000;
    host events = thermo drain, dump, end of run; S must be a multiple of K, else use a tail graph)
for each segment:
    device-to-device snapshot of (x, v, img, gid, step counter)        # for overflow recovery
    launch graph S/K times on one stream
    one cudaStreamSynchronize
    read DeviceStatus (one small memcpy)
    if status.nan_detected: write emergency dump, abort with clear message (step, atom gid)
    if status.overflow:
        restore snapshot; grow capacity ×1.5; rebuild neighbor structure; re-capture graph;
        re-run the same segment (log a warning with new capacity)
        continue
    drain thermo_ring → thermo CSV (convert to physical observables on host in double)
    if dump due: gather x (and optional per-atom quantities) in gid order → extxyz
```

### 6.3 Inside one time step (M4 form, before M6 fusion)
```
K1 integrate_first_half: v += ½dt f/m; x += dt v; wrap; update img; write pos4;
   compute |x − x_ref|² per atom; block-reduce max; atomicMax into status.max_disp2_bits;
   if max > (skin/2)²: set status.need_rebuild (done by the last block or by K2 prologue)
K2 rebuild_if_needed: [all rebuild kernels start with: if (!status.need_rebuild) return;]
   (a) compute cell index per atom, (b) histogram + exclusive scan (CUB),
   (c) build permutation by cell order (Morton-ordered cells), (d) permute ALL per-atom arrays
       into second buffers and swap pointers (x, v, img, pos4, f, gid, slot_of_gid),
   (e) build neighbor list, (f) x_ref = pos4, clear need_rebuild, clear max_disp2, ++rebuild_count
   NOTE: pointer swaps inside a captured graph are not possible → use the "copy back" variant
   (permute into scratch, then copy scratch→primary) OR fixed double buffers with an index flag
   read by kernels from device memory. Choose one, document in DESIGN.md, measure cost.
K3 force<EV>: EV = this step needs energy/virial (thermo/dump step) — two graph variants or a
   device-side flag read at kernel start; prefer device flag `step % thermo_every == 0`.
K4 integrate_second_half: v += ½dt f/m; if thermo step: block partial sums of KE, PE, virial →
   write to thermo_ring[record] via a deterministic two-stage reduction (no float atomics).
   increment device step counter.
```
- **Conditional graph nodes (CUDA ≥ 12.4)** may replace the early-exit pattern for K2 as an optimization in M4; the early-exit version must exist first and stay as the fallback.

---

## 7. Milestones (each = one branch, one PR, one report)

Every milestone PR must satisfy **§9 Git/GitHub requirements**. Acceptance criteria are hard; if one cannot be met, the PR must say so explicitly with measurements and a proposed alternative — do not silently relax.

---

### M0 — Repository bootstrap
**Goal:** empty but complete skeleton; builds; CI runs.

Tasks:
1. Create repo `TileMD` (private is fine). Add the reviewer's access (the user will attach it in Claude's session; just tell the user the repo URL).
2. Commit this plan as `docs/PLAN.md`; create `DECISIONS.md`, `DESIGN.md` (headings only), `CHANGELOG.md`, README skeleton.
3. CMake project: options `TILEMD_ARCHS` (default `80;90`), `TILEMD_BUILD_TESTS` (ON), Release flags `-O3 --use_fast_math` **OFF by default** (fast-math changes numerics; allow only per-kernel, documented). Enable `-lineinfo` always (for Nsight).
4. GoogleTest via FetchContent; one CPU test and one GPU test (`SKIP` if no GPU).
5. `scripts/check.sh` (clang-format check, configure, build, run CPU tests) and `scripts/gpu_check.sh` (all tests incl. GPU, writes log to `reports/logs/`).
6. GitHub Actions `ci.yml`: Ubuntu runner, container `nvidia/cuda:12.4.1-devel-ubuntu22.04`, runs `scripts/check.sh` (compile only for CUDA; GPU tests are skipped in CI).
7. `.github/pull_request_template.md` (Appendix B), `.gitignore` (Appendix A), `.clang-format`.
8. `bench/scripts/collect_env.py`: records GPU name, driver, CUDA version, SM clocks (`nvidia-smi -q -d CLOCK`), ECC, CPU, hostname hash, git commit hash, build flags → JSON.

Acceptance: CI green; `scripts/check.sh` passes locally; README explains build in ≤ 10 commands.
Deliverables pushed: all skeleton files; `reports/M0.md` (short).

---

### M1 — Baselines (can run in parallel with M2)
**Goal:** the numbers to beat, measured on the same GPU, with traces.

Tasks:
1. **GPUMD** (user's fork `WhiteCrosstheRiver/GPUMD`, record exact commit hash): write `bench/inputs/gpumd/` inputs for the LJ melt (§4.4) at all four N; argon metal-unit mapping; NVE; thermo output interval 1000 steps.
2. **LAMMPS** (latest stable, record version) with KOKKOS CUDA: `bench/inputs/lammps/in.lj.kk` equivalent of `bench/in.lj`; run with `-k on g 1 -sf kk -pk kokkos newton off neigh full` and also `neigh half newton on` (record both); also the GPU package variant if available (optional).
3. Timing protocol (§8). Report atom-steps/s per system size; median of 3 runs.
4. Nsight Systems traces (`nsys profile --stats=true`) for N = 32k and 2M for both codes, 2000 steps: commit **only** the text/CSV stats exports (kernel time table, API calls incl. memcpy counts per step), not the `.nsys-rep` binaries.
5. Confirm or refute (with trace evidence) the GPUMD per-step host sync and count kernels per step.
6. Record LAMMPS thermo (PE, KE, P) for the 32k system over 10 000 steps → `tests/validation/lammps_ref/` (used by M3 validation).

Acceptance: `bench/results/<date>_<gpu>_M1_gpumd.json`, `..._M1_lammps.json`; `reports/M1.md` with table: N vs atom-steps/s for GPUMD, LAMMPS-KOKKOS (both list modes); time-per-step breakdown; kernels per step; host syncs per step.

---

### M2 — CPU reference oracle (parallel with M1)
**Goal:** a deliberately simple, obviously correct implementation used to generate fixtures. Speed irrelevant.

Pseudocode:
```
for each atom i:
    for each atom j ≠ i:            # O(N²) brute force for N ≤ 4000; cell list variant for larger
        d = x_i − x_j; apply minimum image (orthorhombic)
        r2 = |d|²; if r2 ≥ rc2(type_i, type_j): continue
        s, u = lj(r2, types)        # exactly §4.1, in double precision
        f_i += s·d; e_i += ½u; W += ½ d ⊗ (s·d)
```
Fixtures to generate (extxyz + JSON of reference forces/energies/virial, double precision):
1. `fcc_500`: perfect fcc, ρ* = 0.8442 (forces ≈ 0; tests cancellation).
2. `melt_4000`: equilibrated LJ melt snapshot (from LAMMPS or the reference itself).
3. `boundary_500`: atoms placed within 1e-3σ of all box faces/edges/corners.
4. `cutoff_500`: pairs placed at r_c − 1e-4 and r_c + 1e-4.
5. `twotype_2000`: two types, distinct ε, σ, r_c per pair.
6. `dilute_1000`: gas-like, density 0.05 (few neighbors; tests empty lists).
7. `dense_2000`: density 1.2 (many neighbors; tests capacity).

Acceptance (gate **G1-ref**): reference reproduces LAMMPS forces on `melt_4000` to ≤ 1e-10 relative (double); fixtures committed (≤ 1 MB each); `reports/M2.md`.

---

### M3 — GPU v0: simple and correct
**Goal:** a correct GPU engine with the conventional design (thread-per-atom, neighbor list). This is the correctness baseline for all later optimizations.

Pseudocode:
```
build_cells:
    cell_size ≥ rc_max + skin per dimension; ncell_x = floor(Lx / cell_size) (≥ 3, else error: box too small)
    cell_of_atom[i] = linear index from wrapped float position
    counts via atomicAdd on int (integer atomics are deterministic in final value)
    CUB exclusive scan → cell_start
    fill cell_atoms in a deterministic order: stable sort atoms by cell index (CUB radix sort on keys=cell, values=i)
build_neighbor_list (thread per atom i):
    for each of the 27 neighboring cells (with periodic wrap):
        for each atom j in that cell, j ≠ i:
            d2 = |min_image(pos4_j − pos4_i)|²
            if d2 < (rc + skin)²: append j to i's list (count < max_neigh else set overflow)
    neighbors in deterministic order (cell order, then atom order within cell)
force_lj<EV> (thread per atom i):
    load pos4_i; f = 0; e = 0; w[6] = 0
    for k in 0..nn[i]−1:
        j = nl[k*N + i]; load pos4_j
        d = min_image(pos4_i.xyz − pos4_j.xyz); r2 = |d|²
        if r2 < rc2[ti][tj]: s,u = lj(...); f += s·d; if EV: e += ½u; w += ½ d⊗(s d) (6 comps)
    write f; if EV: write per-atom e, w to scratch → deterministic block reduction later
integrate (velocity Verlet, §4.3), thermo every `thermo_every` steps with deterministic reductions
```
No spatial sorting yet, no graphs yet (plain launches + host sync each step is acceptable in M3 only).

Tests (gate **G1**):
- T1 Force accuracy vs every fixture: `max_i |F_gpu − F_ref| / rms(|F_ref|) ≤ 1e-5`; PE relative error ≤ 1e-6; virial relative error ≤ 1e-5. (`fcc_500`: absolute max force ≤ 1e-4.)
- T2 Momentum: `|Σ_i F_i| ≤ 1e-4 · rms(|F|) · sqrt(N)`.
- T3 Short NVE vs LAMMPS (32k LJ melt, same initial state): PE/KE/P curves agree within statistical noise over 10 000 steps; report means ± standard error of both.
- T4 NVE drift: `lj_shift = force`, 32k atoms, dt = 0.005, 10⁶ steps (may run overnight): report total-energy drift slope (per atom per τ). Also run LAMMPS with identical settings and report its slope. Pass: TileMD slope ≤ 2× LAMMPS slope.
- T5 RDF g(r) averaged over 50 000 steps vs LAMMPS: L1 difference < 0.02.
- T6 Determinism: two runs, 10 000 steps → bitwise identical final x, v (hash).
- T7 Rebuild correctness: run with `rebuild_every_step = true` vs skin-triggered for 1000 steps → max position difference ≤ 1e-6σ (should be bitwise if neighbor order is identical; document).
- T8 Overflow: force `max_neigh = 8` → detects overflow, grows, and final result identical to a run with correct capacity.

Deliverables: code, tests, `reports/M3.md` with all test tables, first performance numbers (expected: comparable to or slower than GPUMD — fine).

---

### M4 — Performance foundation
**Goal:** remove all overhead that is not physics. Target: **≥ LAMMPS-KOKKOS** for N ≥ 256k (gate G2); **≥ 5× GPUMD at 32k** (gate G3).

Tasks (each as its own commit, each benchmarked; record the gain of each in the report):
1. **Spatial sorting at every rebuild:** order atoms by Morton-ordered cell index (stable); permute all per-atom arrays; maintain `gid`/`slot_of_gid`. Output stays in gid order (test T9: dumps identical to M3 dumps after reordering, up to float noise — or bitwise if neighbor order preserved).
2. **Device-side rebuild decision** (§6.3 K1/K2), early-exit rebuild kernels; remove every per-step device→host copy.
3. **CUDA Graphs**: capture one step; instantiate K-step unrolled graph (sweep K ∈ {1, 10, 50, 200}); segment loop (§6.2) with snapshot-based overflow recovery.
4. **EV templating:** energy/virial only on thermo steps (device flag). Per-atom virial only on dump steps that request it.
5. **Deterministic thermo reductions** into `thermo_ring` (no host interaction until segment end).
6. **Force kernel tuning:** `__launch_bounds__`, `__ldg`/read-only path for `pos4`, block size sweep {64, 128, 256}; type parameters in `__constant__`.
7. **Neighbor threading for small N** (optional): T threads per atom (T ∈ {1, 2, 4, 8}) with warp-shuffle reduction; enable automatically when N < ~100k (measure crossover).
8. **Conditional graph nodes** for the rebuild branch (optional optimization; keep fallback).

Pseudocode for the displacement check (K1 epilogue):
```
d2 = |pos4_new − x_ref|² (min image)  → block max via CUB BlockReduce
if threadIdx.x == 0: atomicMax((int*)&status.max_disp2_bits, __float_as_int(block_max))  # valid for non-negative floats
rebuild kernel prologue: if (__int_as_float(status.max_disp2_bits) > (skin/2)²) need_rebuild = 1
```

Acceptance: all M3 tests still pass (re-run, attach log); `reports/M4.md` with a waterfall table "optimization → atom-steps/s at 32k / 256k / 2M / 16M" and Nsight stats showing 0 memcpy per step and kernels per step; benchmark JSON committed.

---

### M5 — Cluster-pair tiles (SIMT)
**Goal:** replace per-atom neighbor lists by cluster-pair lists and 8×8 dense tiles; cut list traffic ~8× and create the tile structure needed for M7.

Pseudocode:
```
build_cluster_grid (at rebuild):
    choose column side a so that mean atoms per column ≈ 32 (= 4 clusters); columns on x–y grid
    assign atoms to columns; within column sort by z (stable); pad column to multiple of 8 with dummy atoms
    cluster c = 8 consecutive slots; compute bbox[c] (ignore dummies)
build_cluster_pair_list (thread or warp per i-cluster):
    for each column within (rc + skin) in x–y (periodic, with shift index):
        for each j-cluster in that column whose z-range overlaps [lo_z − (rc+skin), hi_z + (rc+skin)]:
            if bbox_distance(ci, cj, shift) < rc + skin: append (cj, shift)
    store CSR; count statistics: pairs inside cutoff / pairs computed ("tile efficiency")
force_tile<EV> (one warp per i-cluster):
    lanes 0..31 = (i_local ∈ 0..7) × (j_sub ∈ 0..3)
    load the 8 i-atoms into registers (each lane holds its i_local atom)
    for each group of 4 j-clusters in the list:
        each lane loads/broadcasts its j-cluster's 8 atoms (shared memory or shuffle) applying shift vector
        lane loops over 8 j atoms: d2 → mask (cutoff, dummy, self) → s,u → accumulate f_i (and e, w if EV)
    reduce f_i over the 4 j_sub lanes with warp shuffles (fixed order ⇒ deterministic)
    write f for the 8 real i atoms
```
Measure and report: tile efficiency (expected ~40–60%), bytes per atom-step from Nsight Compute (`dram__bytes`), L1/L2 hit rates, achieved occupancy.

Acceptance: all correctness tests pass; speed ≥ M4 at N ≥ 256k (if not, report why with Nsight Compute evidence and keep both paths selectable); `reports/M5.md`.

---

### M6 — Fused single-kernel time step
**Goal:** one kernel launch per time step ("one pass"): no force array, positions double-buffered.

Pseudocode (leapfrog, §4.3):
```
buffers: pos4[A], pos4[B]; x, v in double (single copy — each atom updates only itself)
fused_step<EV>(read = pos4[cur], write = pos4[next]):   (one warp per i-cluster, as M5)
    compute f_i from pos4[cur] for the 8 i atoms (tile loop of M5)
    lanes owning an i atom:
        v_new = v_old + dt · f_i / m              (+ Langevin terms if NVT, Philox keyed by gid, step)
        if EV: KE contribution uses ½(v_old + v_new); PE, virial as before (block partials)
        x_new = x + dt · v_new; wrap; update img
        write x, v (double) for own atom; write pos4[next] for own atom
        displacement check vs x_ref → block max → status
    step counter++ (one thread)
host graph: [fused_step(cur=A,next=B)] [rebuild_if_needed on B] [fused_step(cur=B,next=A)] ...
    (the graph alternates buffers explicitly, so no pointer swaps are needed)
```
Correctness subtlety to handle and document: `x`/`v` (double) for atom i are read and written only by the thread that owns i ⇒ safe; neighbors are read only from `pos4[cur]` ⇒ safe. Rebuild must permute both pos4 buffers consistently (only the freshly written one is needed).

Acceptance: correctness tests (T1 via a forces-only debug kernel, T3–T8 via trajectories) pass; leapfrog vs velocity Verlet trajectories agree to round-off for 100 steps when started consistently; speed ≥ M5; `reports/M6.md` including bytes per atom-step vs the §1.2 roofline estimate (~60–110 B/atom-step target).

---

### M7 — Tensor-core "pair-attention" tile (experimental, gated)
**Goal:** move the matrix parts of the pair computation onto tensor cores. Kept only if gate G4 passes.

Mathematical form (per tile of 16 i-atoms × 8 j-atoms, coordinates relative to the i-tile center, j shifted by its periodic image):
```
Scores:   A_i = [x_i, y_i, z_i, |r_i|², 1, 0, 0, 0]          (16 × 8)
          B_j = [−2x_j, −2y_j, −2z_j, 1, |r_j|², 0, 0, 0]ᵀ  (8 × 8)
          D = A · B   ⇒  D_ij = |r_i − r_j|²                 (one mma.m16n8k8 TF32, ×3 for 3×TF32)
Kernel:   S_ij = mask(D_ij < rc², not dummy, not self) · s(D_ij)   — elementwise ON the accumulator
          fragment registers (FlashAttention trick); U_ij likewise if EV
Values:   V_j = [x_j, y_j, z_j, 1, 0, 0, 0, 0]               (8 × 8)
          P = S · V  ⇒  P_i = [Σ s x_j, Σ s y_j, Σ s z_j, Σ s]
Force:    F_i = r_i · (Σ_j S_ij) − (Σ_j S_ij r_j)           (= Σ_j S_ij (r_i − r_j))
Virial (EV only): second value matrix with columns [x_j², y_j², z_j², x_j y_j, x_j z_j, y_j z_j, 1, 0] and
          the expansion of Σ_j S_ij (r_i − r_j)⊗(r_i − r_j); energy via U · 1.
```
3×TF32 split: for each operand value a: `a_hi = tf32_round(a)`, `a_lo = tf32_round(a − a_hi)`; compute `A_hi·B_hi + A_hi·B_lo + A_lo·B_hi`.
Implementation notes:
- Use inline PTX `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` (or CUTLASS `cute` atoms). Document the accumulator → A-operand fragment conversion (register shuffles or a shared-memory round trip); measure both.
- Write `docs/NUMERICS.md`: error budget for d² (cancellation factor (|r|/d)² with tile-relative coordinates), force cancellation, comparison of plain TF32 vs 3×TF32 vs FP32 SIMT on all fixtures.

Acceptance (gate **G4**): force errors vs reference within T1 tolerances with 3×TF32; NVE drift (T4) not worse than M6; speed ≥ 1.3× M6 at N ≥ 256k. If any fails → keep code on a branch, document the result in `reports/M7.md` (a negative result is a valid, valuable deliverable), and do not merge into the default path.

---

### M8 — Generic pair-attention API + EAM
**Goal:** turn the tile machinery into a reusable primitive and prove it on a many-body potential.

API sketch (C++ templates, no virtual calls in kernels):
```
struct Potential concept:
    static constexpr int num_passes;
    per pass p:
        pair_scalar_p(r2, type_i, type_j, atom_attrs_i, atom_attrs_j) → (s_force, s_energy, s_value...)
        value_columns_p(j)          → which per-j quantities are aggregated (1, r_j, r_j⊗r_j, custom)
        atom_update_p(i, aggregates) → per-atom outputs (e.g., ρ_i → F(ρ_i), F'(ρ_i))
LJ   = 1 pass:  s = s_LJ(r²); values [r_j, 1]
EAM  = 2 passes: pass 1 values [1] with scalar f_density(r) → ρ_i; atom_update: F'(ρ_i)
                 pass 2 scalar (F'(ρ_i) + F'(ρ_j))·ρ'(r)/r + φ'(r)/r, values [r_j, 1]
```
Validation: EAM Cu with a LAMMPS-shipped `funcfl`/`setfl` file (e.g. `Cu_u3.eam`, metal units → this milestone introduces metal units, add to D3 in DECISIONS.md); forces vs LAMMPS `pair eam` on fixtures (≤ 1e-5 relative), NVE drift, benchmark vs LAMMPS-KOKKOS EAM.

Acceptance (gate **G5**): LJ through the generic API has ≤ 5% overhead vs the specialized M6/M7 kernel; EAM passes correctness; `reports/M8.md`. This API is the entry point for NEP in stage 2 (NEP functional form unchanged).

---

## 8. Benchmark protocol (all milestones)

- Build: Release, `-lineinfo`, no fast-math unless documented.
- Before each run: record environment via `collect_env.py`. Do not run other jobs on the GPU. Note if clocks could not be locked.
- Procedure per (code, N): load/generate system → 1000 warm-up steps → timed run of `T` steps (32k: 100 000; 256k: 20 000; 2M: 5000; 16M: 1000) with thermo every 1000 steps → timer = CUDA events around the step loop (TileMD) or the code's own loop timer (GPUMD/LAMMPS "Loop time") — document which.
- Metric: `atom_steps_per_s = N · T / t_loop`. Repeat 3×, report median and min/max.
- Also report: ns/day equivalent for argon (τ = 2.156 ps), peak GPU memory, rebuild count.

Result JSON schema (`bench/results/<YYYY-MM-DD>_<gpu>_<milestone>_<code>.json`):
```json
{
  "code": "tilemd|gpumd|lammps-kokkos",
  "code_version": "git hash or release",
  "milestone": "M4",
  "env": { "...": "output of collect_env.py" },
  "system": { "name": "lj_melt", "N": 256000, "rho": 0.8442, "rc": 2.5, "skin": 0.3, "dt": 0.005 },
  "settings": { "list": "full", "graph_unroll": 50, "block_size": 128, "...": "..." },
  "runs": [ { "steps": 20000, "t_loop_s": 1.234, "atom_steps_per_s": 4.1e9, "rebuilds": 812 } ],
  "median_atom_steps_per_s": 4.1e9
}
```
`bench/scripts/plot_bench.py` produces `reports/figures/bench_<milestone>.png` (atom-steps/s vs N, all codes, log–log).

---

## 9. Git / GitHub requirements (mandatory)

### 9.1 Branching and PRs
- Default branch `main` is always green (builds, CPU tests pass). Protect it: PRs only.
- One branch per milestone (names in §0). Sub-branches allowed (`m4-perf-foundation/graphs`), merged into the milestone branch.
- **One PR per milestone**, opened as draft early; mark "ready for review" when acceptance criteria are met or explicitly reported as unmet.
- After a gate passes and the PR merges: tag `v0.<milestone>.0` (e.g. `v0.4.0`), update CHANGELOG.

### 9.2 Commits
- Small, single-purpose commits; message format (Conventional Commits): `feat(neighbor): morton-ordered spatial sort`, `fix(force): ...`, `perf(force): ...`, `test: ...`, `bench: ...`, `docs: ...`, `refactor: ...`, `build: ...`.
- Every **perf** commit message must include the measured before/after (e.g. `perf(force): read-only pos4 loads (256k: 2.1e9 → 2.4e9 atom-steps/s, H100)`).
- Each commit must build. `scripts/check.sh` must pass before push.

### 9.3 What every milestone PR MUST contain (reviewer will reject otherwise)
1. Code + tests for the milestone.
2. `reports/M<k>.md`: what was built, test result tables (T1–T9 as applicable), benchmark tables, Nsight summaries, deviations from this plan, open questions.
3. Benchmark JSON files in `bench/results/` and the plot in `reports/figures/`.
4. GPU test log from `scripts/gpu_check.sh` in `reports/logs/M<k>_gpu_check.log` (text, ≤ 1 MB).
5. Nsight Systems/Compute **text or CSV exports** (not binary reports) in `reports/profiles/M<k>/`.
6. Updated `docs/PLAN.md` status table and, if behavior/design changed, `docs/DESIGN.md`; any deviation from §2 recorded in `docs/DECISIONS.md` with evidence.
7. CHANGELOG entry.
8. Filled PR template (Appendix B).

### 9.4 What must NOT be committed
- Build directories, binaries, `.nsys-rep`/`.ncu-rep` files, trajectories or dumps > 5 MB, LAMMPS/GPUMD source trees (reference them by version/commit hash instead), credentials, cluster hostnames/usernames (hash them).
- If a large artifact is essential, attach it to a GitHub Release, not to the git history.

### 9.5 Communication on GitHub
- Questions needing a decision: open an issue with label `decision-needed` using `.github/ISSUE_TEMPLATE/decision.md` (context, options, measurements, recommended option). Continue with the recommended option unless the decision is irreversible.
- Request review by commenting on the PR: `@reviewer ready for review: M<k>` and summarizing in ≤ 10 lines what to look at first.
- Reviewer findings will be posted as PR review comments with severity tags: `[blocker]`, `[major]`, `[minor]`, `[nit]`. All `[blocker]` and `[major]` must be resolved or answered before merge.
- Keep a running `reports/REVIEW_LOG.md`: one line per review round (date, PR, outcome).

---

## 10. Execution plan

### 10.1 Right now
1. M0 bootstrap (repo, build, CI, templates, this plan committed).
2. Start M1 baselines on H100 (GPUMD + LAMMPS-KOKKOS) — needs only input files, not TileMD code.
3. Start M2 CPU reference and fixtures.

### 10.2 Sequential chain (each depends on the previous gate)
M0 → M2 (G1-ref) → M3 (G1) → M4 (G2, G3) → M5 → M6 → M7 (G4, optional) → M8 (G5).

### 10.3 Can run in parallel
| Parallel track | Can start after | Feeds into |
|---|---|---|
| M1 baselines | M0 | gates G2/G3, M4 targets |
| M2 reference + fixtures | M0 | M3 tests |
| Validation harness (NVE drift, RDF, pressure scripts vs LAMMPS) | M0 (+ M1 LAMMPS refs) | M3 onwards |
| Benchmark harness (`run_bench.py`, `collect_env.py`, plotting) | M0 | all milestones |
| Cluster-pair list builder (M5 data structure) on CPU first, unit-tested | M3 | M5 |
| Triton/Python prototype of the 8×8 tile and of the M7 3×TF32 math (numerics only) | M2 | M5, M7, NUMERICS.md |
| EAM reference (CPU) + fixtures | M2 | M8 |

---

## 11. Gates summary

| Gate | Criterion |
|---|---|
| G1-ref | CPU reference matches LAMMPS forces ≤ 1e-10 relative (double) |
| G1 | GPU forces ≤ 1e-5 relative vs reference; NVE drift ≤ 2× LAMMPS; RDF, PE, P match LAMMPS; deterministic |
| G2 | ≥ LAMMPS-KOKKOS atom-steps/s on the same GPU for N ≥ 256k (stretch: ≥ 3×10⁹ on H100 at 2M) |
| G3 | ≥ 5× GPUMD at N = 32k |
| G2+/G3+ | M5/M6 each ≥ previous milestone at their target sizes, or documented reason |
| G4 | Tensor-core path: accuracy within G1, drift unchanged, ≥ 1.3× M6 |
| G5 | Generic API ≤ 5% overhead for LJ; EAM correct vs LAMMPS |

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Graph capture incompatible with rebuild permutation | Fixed double buffers + device-side buffer index; or copy-back permutation; measure |
| FP32 shadow positions lose precision in very large boxes (16M atoms, L ≈ 270σ) | Use cell-relative or tile-relative coordinates in force kernels; document error at L = 270σ |
| Tile efficiency too low (many masked pairs) | Tune column size; bbox pruning; report efficiency per N |
| Tensor-core path gives no gain (LJ elementwise part dominates) | G4 is optional; negative result documented |
| Benchmarks noisy on shared cluster nodes | Exclusive node allocation; 3 repeats; record clocks |
| Scope creep (NPT, triclinic, NEP) | Not in stage 1 (§1.4); file as issues labeled `stage-2` |

---

## Appendix A — `.gitignore`
```
build/
build-*/
cmake-build-*/
*.o
*.so
*.a
*.nsys-rep
*.ncu-rep
*.sqlite
*.qdrep
__pycache__/
*.pyc
.venv/
*.dump
*.lammpstrj
dumps/
scratch/
.DS_Store
.idea/
.vscode/
```

## Appendix B — `.github/pull_request_template.md`
```
## Milestone
M<k> — <name>

## What changed
- ...

## Why (link to PLAN.md section)
- ...

## How it was tested
- [ ] scripts/check.sh passes
- [ ] scripts/gpu_check.sh passes (log: reports/logs/M<k>_gpu_check.log)
- [ ] Correctness tests T1–T9 (as applicable) — table in reports/M<k>.md

## Results
| N | atom-steps/s (this PR) | previous | GPUMD | LAMMPS-KOKKOS |
|---|---|---|---|---|

## Deviations from PLAN.md / DECISIONS.md changes
- none / ...

## Open questions for reviewer
- ...

## Checklist
- [ ] PLAN.md status updated
- [ ] DESIGN.md / NUMERICS.md updated if relevant
- [ ] CHANGELOG updated
- [ ] Benchmark JSON + plot committed
- [ ] No binaries / large files committed
```

## Appendix C — `.github/ISSUE_TEMPLATE/decision.md`
```
---
name: Decision needed
labels: decision-needed
---
## Context
## Options (with measurements if available)
## Recommended option and why
## Is it reversible? (yes/no)
## Proceeding with: <option> (unless told otherwise)
```

## Appendix D — Input file sketch (`*.in`)
```
units          lj
box            fcc 40 40 40 0.8442      # lattice nx ny nz density   (or: structure file.extxyz)
types          1
lj             0 0  epsilon 1.0  sigma 1.0  cutoff 2.5
lj_shift       none                     # none | energy | force
mass           0 1.0
velocity       1.44 seed 12345
skin           0.3
timestep       0.005
ensemble       nve                      # nve | langevin T gamma seed
thermo_every   100
dump_every     0                        # 0 = off
dump_virial    0                        # 1 writes per-atom virial on dump frames
graph_unroll   50
run            10000
```
