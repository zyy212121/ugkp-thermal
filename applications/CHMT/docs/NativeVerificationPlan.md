# CHMT production-native verification plan

## Status and review gate

This is a proposed acceptance plan, not a runnable test suite or a certificate.
The production-native case generator, runner and result checker described below
are **not implemented in this candidate**. Review this plan together with the
solver before implementing that missing harness. Return code findings, plan
corrections, required instrumentation and the intended commands to the owner;
wait for approval before changing the implementation or starting native runs.

The submitted component tests and native CPU-driver test remain useful prior
evidence. `devtools/of_coupling` runs actual OpenFOAM finite-volume equations with
CHMT components, but its gas adapter and manual macro loop do not execute the
production CUDA backend or `advanceCoupledWindow`. Its results must never fill
the production-native acceptance cells below. The existing `adapter/chmt-adapter`
supports isolated mathematical/standalone cases, not the missing coupled
Multirate regression suite. Historical CUDA status is NOT RUN, not PASS.

## Workspace and immutable inputs

- Repository: `https://github.com/zyy212121/ugkp-thermal.git`.
- Formal checkout: `/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal`.
  Inspect only; do not reset, clean, overwrite, build in, or merge into it.
- Development area: `/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test`.
  Inspect its status first. Use an isolated worktree at the exact review commit,
  at a new unused path associated with this development area. Preserve every
  existing dirty/untracked file. Do not repurpose or delete an existing worktree.
- Review the branch against its actual publication base, initially
  `575b7dd3d30cc648142914d6e81765a43e04222c`; record the reviewed commit and merge
  base. Restrict proposed changes to `applications/CHMT`. Preserve all other
  applications, `common`, root tests and README files.
- Keep generated cases, binaries and evidence outside tracked source. Record
  source/input hashes, compiler/OF/CUDA versions, GPU model, precision, commands,
  return codes and raw outputs. Never overwrite input or reference results with
  solver-produced values. Do not merge, push additional changes or open a PR
  merely because a verification stage succeeds.

## What counts as a native production run

Build the actual `CHMT` executable using Foundation OpenFOAM 10, nvcc and a real
CUDA device. Use the existing separate host material library and CUDA backend:

```sh
# First source the installed Foundation OpenFOAM 10 environment.
# Set CHMT_CUDA_ARCH to an architecture supported by this nvcc and actual GPU.
# Set CHMT_BUILD_DIR to a fresh absolute directory outside the checkout.
CHMT_SPECIES=S0,S1 bash applications/CHMT/Allwmake
```

The environment variables above must actually be exported or supplied with the
command; no architecture is presumed. S0/S1 are explicitly synthetic verification
species. Record the actual build/device identity and dependencies. No substitute
headers, no invented device identity, no CPU gas fallback, and no mocked sparse
solve can satisfy this stage. Missing prerequisites produce BLOCKED/NOT RUN.

Each coupled acceptance case must execute the normal frontend with
`executionMode Multirate`, a real gas inventory and 3-D solid fvMesh. Trace and
verify this production call path:

`CHMT.C -> advanceCoupledWindow -> beginGasWindow / advanceGasMicrostep ->
CpuMaterialDriver::advanceCandidate -> audits -> commitGasWindow`.

Verify both CUDA stages execute, gas stays on the GPU, CPU conduction uses the
native matrix solve, and packet ownership changes only through the actual
production transaction. Existing `execution-mode.json`, `multirate.json`,
`windows.csv`, raw inventories and stage-geometry outputs supply part of this
evidence. Identify any missing observability in review; do not infer a code path
merely from a filename. Minimal noninvasive test instrumentation may be proposed
for events/replays that current output cannot distinguish.

## Missing harness deliverables, after plan approval

Implement under `applications/CHMT/devtools/native/` and/or a CHMT-local cases
directory, without changing unrelated test packages:

1. A deterministic input generator for genuine `CHMT` cases: full gas and solid
   meshes, fields, boundary dictionaries, `chmtProperties`, `controlDict`,
   `fvSchemes` and `fvSolution`, plus a manifest of physical parameters.
2. A runner that builds/locates the genuine executable, performs preflight and
   native `checkMesh`, launches all prescribed cases, records the exact exit
   status, and fails rather than silently skipping an unavailable required case.
3. An independent checker reading raw computed fields, transfer budgets and
   actual stage sweeps. It must not integrate a replacement model, generate
   future solution values, prescribe wall heat/mass histories for coupled
   acceptance, or substitute the verification gas adapter.
4. A machine-readable result per case: PASS, FAIL, BLOCKED or NOT RUN, with
   thresholds fixed before runs and links to logs/raw data. A negative test must
   report the expected rejection plus state-preservation evidence.

These are proposed deliverables, not currently available commands. Audit the
current parser and output contracts before promising a ready-to-run harness.

## Small case matrix

### N0: Native build and production-path smoke

Build/link all production libraries and the frontend, then run a fixed,
nonreactive gas/solid case with at least 20 accepted macro windows and at least
four gas microsteps per ordinary window. Use positive temperatures/densities,
synthetic equal-property S0/S1 gas, no particles, SST or radiation initially.
Require genuine CPU matrix solves and CUDA execution, no unexpected rejection,
and synchronized output/checkpoint state. Check that gas microsteps and material
RHS/linear-solve counts are distinct. Count executed work including replays.

### N1: Fixed geometry, equilibrium and implicit-conduction accuracy

First run a uniform-temperature, no-source closed coupled equilibrium; it should
preserve the state within accumulation-scaled floating-point tolerances. Then
exercise a 3-D solid thermal mode on the production path with the relevant
interface heat exchange disabled by a supported physical configuration. Compare
against an independently derived discrete backward-Euler eigenmode for constant
properties, using actual mesh/discrete operator and actual material substeps.
If this configuration cannot be expressed by the existing frontend, report a
harness/configuration gap; do not replace this case with the CPU-driver test.

Next enable full two-way gas/solid heat transfer with hot gas and a transverse
solid temperature gradient. Require evolving gas fields and wall response,
nonzero opposite-signed heat exchange, global energy closure and refinement.
An analytic Dirichlet solid problem validates conduction only; it does not
certify the coupled boundary flux. For variable heat capacity, independently
evaluate formation-inclusive energy from the computed inventories and compare
to an over-resolved numerical reference, without fitting the reference to the
coarse run.

### N2: Actual CUDA moving-grid GCL

Use a supported prescribed motion with a uniform gas state and compatible
boundaries; execute the native production gas path and record both stages.
Check every conserved gas component, all cell volumes and oriented face sweeps,
not just temperature. A gas-only Standalone run is allowed for this isolated
GCL gate, but does not satisfy any Multirate coupling gate.

### N3: Dry, nonuniform 3-D recession through Multirate

Use conformal tetrahedra with initially planar shared interface, initially
192 gas and 96 solid cells. The verified tetra-mesh construction can be reused
as input preparation, but the verification adapter must not execute the case.
Check native mesh quality before and after motion, retaining native mesh files
or a lossless export of accepted points/topology for that purpose.

Suggested starting domain: gas x=0..20 mm, solid x=-4..0 mm, transverse dimensions
10 mm by 10 mm. Start gas at 700 K and 100 kPa; solid at
`500 + 40*y/0.01 + 20*z/0.01 K`. Use the documented synthetic material/Arrhenius
parameters from `devtools/of_coupling/NATIVE_COUPLING.md` only after verifying
that all are expressible by the production model parser. No prescribed future
wall temperature, recession history, mass rate or heat-flux response is allowed.

Require actual nonzero transfer, two-axis nonuniform recession, evolving gas and
solid fields, interface pressure work exactly once, and separately measured
external moving-wall/ALE budgets. Advance initially to 0.1 ms. Require at least
25 accepted windows; investigate inability to achieve that rather than silently
shortening the observation. Check physical mass-loss/volume-recession agreement,
bounded carried sweep remainder and valid dense-material porosity throughout.

### N4: Independent gas-step and coupling-window refinement

For N3, initially test gas caps `[4e-7, 2e-7, 1e-7] s` with `H=4e-6 s`; separately
test `H=[4e-6, 2e-6, 1e-6] s` with gas cap `1e-7 s`. They are initial requested
values, not promised stable steps: use actual accepted steps in the report.
Do not claim independent gas refinement if CFL clips all three onto the same
trajectory. Keep other controls, output times, reconstruction and material
accuracy fixed; demonstrate material error is smaller with a separate tightening
check. Include both full-field endpoint and peak trajectory norms at common
physical times, not only a selected probe or recession scalar.

For a resolved first-order coupling signal, target fine/coarse successive
difference ratio near 0.5; use `<0.75` as the preliminary acceptance ceiling and
require monotone reduction. Explain any different expected order from the actual
scheme. Reject a false convergence claim when differences are below roundoff,
when two levels secretly share actual clocks, or when normalization changes.
Review the threshold before running; do not loosen it after seeing results.

### N5: Native synchronized restart, replay and rejection

Split N3 at an accepted macro synchronization. Restore the actual binary
CHMTCP2 checkpoint into the genuine executable and compare the next window and
final endpoint against uninterrupted progression on the same hardware/build.
Compare gas/SST/solid/film inventories, mesh points, stage/counter state,
commitSequence, physical budgets and signed geometric carry. There is no
supported pending-window resume. Require exact agreement for integer/discrete
state; target bitwise floating-point agreement where the execution is
deterministic, otherwise predeclare a justified roundoff-level tolerance and
explain the source of nondeterminism. A loose field norm cannot hide ledger or
counter differences.

Exercise a recoverable prediction/geometry mismatch causing an actual production
replay or shortened window, and an unrecoverable native failure after a previously
accepted window. Preserve that accepted CPU/GPU/mesh state, avoid duplicated
packets, and verify the last-accepted checkpoint can resume consistently. Select
a deterministic supported failure configuration in review. If no configuration
can demonstrate rollback observably, propose minimal fault instrumentation and
obtain approval; scripted participants cannot satisfy this gate. Also test the
known incompatible quad-motion case: explicit rejection and unchanged accepted
state are required, and are not evidence that arbitrary moving hex meshes work.

### N6: Supported phase-transition coverage, separately gated

Review whether a genuinely coupled planar supported wet case can reach an
event-aligned birth/dryout under the current geometry and material restrictions.
If feasible, exercise it with the production scheduler and check exactly-once
old-owner consumption and the next-window owner switch. If current restrictions
prevent the requested configuration, report NOT SUPPORTED and specify the
restriction. Host phase tests do not certify this native path. Do not introduce
moving curved-offset films, porous melting, wet pore outflow, finite-duration
particle contact or topology changes just to increase coverage.

## Independent acceptance accounting

Derive signs and units from the documented production budget contract first.
For each window and the entire run, independently recompute mass/species/elements
and formation-inclusive total energy from actual inventories and exterior fluxes.
Include ALE transport, interface pressure work, outer-wall work, radiation/body
work when enabled, and the declared reduced-film kinetic defect where applicable.
Internal exchanges must cancel; they must not be counted as exterior supply.

Predeclare each normalization scale. Use actual initial/final inventory plus
cumulative absolute exterior throughput; do not normalize a nearly cancelling
energy residual by the tiny net energy change. Use species/element-specific
scales and dimensional absolute floors derived from the smallest resolved
physical transfer. Require no negative inventory beyond documented roundoff,
positive density/temperature/volume, finite EOS recovery, and valid porosity.
Never clip state or relax physical bounds to pass a test.

For these small double-precision cases, initial conservative goals are relative
mass/species/element residual <=1e-10 and energy residual <=1e-8, both per-window
and cumulatively. These are numerical-closure goals, not solution accuracy.
The reviewer must justify them using summation count, epsilon, inventory scales
and solver residuals, and tighten them when warranted. Reject unexplained secular
growth even if the final scalar passes. Keep acceptance thresholds immutable
once runs start; any change requires a documented reason and a full rerun.

For each actual gas stage/cell, compute `Vnew - Vold - sum(oriented face sweep)`
and normalize by local volume. Aim for 512 double eps for small well-conditioned
fixtures, with any larger bound derived explicitly from geometry conditioning
and accumulation count. Check cumulative sweep remainder and packing volume
independently. Uniform-state drift should be <=1e-10 relative on the small GCL
case. Analytic/reference solution tolerances must separately include linear/
nonlinear solver error and the actual spatial/temporal discretization order;
conservation alone does not establish correct transfer physics or convergence.

For performance, report wall time, synchronization/transfer overhead where
measurable, executed and accepted gas microsteps, CPU material/RHS/linear solves,
replays, peak history/storage use and rejected work. Verify constant forcing does
not mechanically require a full material solve for every gas step. Do not claim
GPU speedup, broad stability or engineering readiness from these tiny tests.

## Review and final report

1. First return prioritized code findings with file/line, mechanism, consequence
   and a minimal reproducer; distinguish proven bugs from untested risks.
2. Review feasibility of every planned native case, all input/output gaps,
   proposed thresholds and necessary harness work. Return an adjusted execution
   plan for approval before implementation/runs.
3. After authorized execution, provide a case matrix with PASS/FAIL/BLOCKED/
   NOT RUN, exact build/input provenance, raw evidence, independent error norms,
   budgets, restart/rollback differences and measured cost.
4. Keep native CPU components, the verification adapter, actual CUDA gas runs
   and full production Multirate runs in separate evidence categories. List every
   unrun gate explicitly. No long engineering runs or automatic formal merge.
