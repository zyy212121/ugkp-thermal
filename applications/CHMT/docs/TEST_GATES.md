# CHMT test gates and coverage

Run all available host gates from any directory:

    bash applications/CHMT/devtools/check_host.sh

A passing host gate is **not** a native OpenFOAM build, CUDA build/device run,
full CPU-material/GPU-gas application result, or physical validation.

## Scope guard and provenance

`check_core.sh` defaults to upstream commit
`db455604156419a9e20b13f1b43694fde33be6c8`, the parent of the original additive
CHMT feature. Against this baseline, all changes must be additions under
`applications/CHMT/`; README additions and changes outside that application are
rejected. Checking against HEAD instead would misclassify edits inside the
already-added application as changes to pre-existing upstream files. An explicit
argument or `CHMT_SCOPE_BASE` overrides the baseline for another reviewed lineage.

The originally delivered core/runtime/gas-mesh/material-film scripts referred to
host test files that were absent from the checkout and its tracked history.
Those entrypoints now execute newly added current-source regressions and the
maintained component suites. This repairs runnable coverage; it does not claim
the unavailable historical suites were recovered or passed.

## Runnable host gates

| Entry point | Executed coverage |
|---|---|
| `devtools/check_core.sh` | Whole-feature additive scope; compiled species identity; variable-cp/formation caloric inversion over inventory scales; reaction stoichiometry/inventory rollback; packet decomposition validation |
| `devtools/check_runtime.sh` | Paired packet/resource atomicity; actual 3-D checkpoint roundtrip, corruption/model rejection and injected fsync failures; particle event/heat/work math; dryout/birth through real cursor, macro audit and film update with scripted participants; interval and frontend suites below |
| `devtools/check_interval_contracts.sh` | Exact overlap histories; microstep identity; owner-level audit; donor prefixes, skipped owners, arithmetic depletion; sparse prepare/publication contracts |
| `devtools/check_multirate_frontend.sh` | Build/source routing and scripted-participant execution of the actual macro controller, clocks, replay, rollback and audit rejection |
| `devtools/check_gas_mesh.sh` | Actual polyhedral geometry/sweeps plus gas ALE free-stream preservation; equal-state face reversal; oblique least-squares gradient; invalid-state rejection |
| `devtools/check_material_film.sh` | Both maintained CPU material and CPU film component suites below |
| `devtools/multirate/check_cpu_material.sh` | Actual caloric, 3-axis transport/reaction, surface-interface and gas-driving coalescing helpers |
| `devtools/multirate/check_cpu_film.sh` | Actual CPU film candidate: lateral transport, energy/pV/work ownership, donor rollback, open-boundary elements and terminal depletion |
| `devtools/multirate/check_sweep_constraints.sh` | Actual constrained 3-D geometry, nonuniform/shared-vertex motion, independent GCL, incompatibility/rank and excluded moving-offset rejection |
| `devtools/multirate/check_cpu_geometry_state.sh` | Actual planar wet-film mass/volume certificates and accepted-stage checkpoint roundtrip |
| `gpu/tests/check_gas_window_host.sh` | Gas wall/immutable waveform/donor/independent-CFL host math and CUDA ownership source structure |
| `devtools/multirate/check_darcy_accuracy.sh` | Actual CPU Darcy operator analytic convergence and finite-difference pressure-Jacobian stability checks; retained CUDA wiring inspected, not executed |
| `devtools/multirate/check_reaction_accuracy.sh` | Actual CPU local reaction accuracy, startup, cross-reaction, thermal-feedback, finite-depletion and transactional-failure checks |

`check_runtime.sh` links `test_restart.cpp` with `--wrap=fsync`: failures before
atomic rename must preserve the old checkpoint; directory-fsync failure after
rename must report uncertified durability while leaving a complete readable file.
No claim is made that a failed post-rename durability check rolls back the rename.

The dryout/birth transaction tests deliberately script the gas producer and
material sequencing. They execute the actual interval cursor, conservation audit,
macro controller and CPU film update across four accepted gas records per window.
They do not execute `CpuMaterialDriver`, CUDA event detection or full CFD.

## Separate native and integrated gates

- `devtools/check_frontend.sh`: actual Foundation OpenFOAM 10 frontend object
  compilation only. It cannot establish link/runtime behavior.
- `devtools/multirate/check_cpu_material_of.sh [case]`: actual native CPU material
  driver compile/link/run using Foundation OpenFOAM 10. No substitute sparse
  solver is permitted. With no case argument, the checked-in native test fixture
  is prepared with real `blockMesh`.
- `devtools/of_coupling/check_native_coupling.sh [evidence-directory]`: actual
  Foundation OpenFOAM 10 central-upwind gas finite-volume equations coupled to
  production CPU material, interface packets and constrained moving geometry.
  Includes free-stream, fixed-interface, nonuniform dry recession, independent
  gas-step/window refinement, native initial/final `checkMesh`, conservation,
  rejection/rollback and synchronized restart. This verification-only adapter
  uses a manual macro loop; it is not the production CUDA gas driver or complete
  application scheduler. The checked-in runner requires every prescribed case.
- `Allwmake`: separate nvcc CUDA backend and wmake CPU material/frontend builds.
  Requires real Foundation OpenFOAM 10, nvcc, selected CUDA architecture and an
  explicit compiled species set.
- Real GPU/native integration acceptance: a gas-containing 3-D case with multiple
  gas microsteps per material window; independent coupling-interval and gas-step
  refinement; actual packet/energy/element/GCL audits; supported phase events and
  replay; uninterrupted-versus-synchronized-restart equivalence; material solve
  counts and measured runtime. Host scripted participants cannot satisfy this.

Missing prerequisites must be reported as blocked, never silently skipped or
counted as passed. Unsupported moving curved-offset wet films, nonconformal or
topology-changing coupling and finite-duration particle contact remain outside
the new window path; tests of explicit rejection are not support for those paths.
