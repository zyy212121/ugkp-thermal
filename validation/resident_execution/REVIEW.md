# Resident execution review and measurements (PR25)

Stacked on corrected PR24. gasUGKP remains the pure-flow application; CHT remains
conjugate heat transfer; CHMT remains conjugate heat/mass transfer and ablation.
No case-specific numerical operator, no alternate CHT application, no FSH or
CHT numerical change is introduced.

## Changes and ownership

* gasUGKP links its existing CUDA resident implementation directly, following
  CHT's single-process pattern. Remove the socket client, wire codec and helper
  process; preserve a typed API, configurable species shape, FMA and probe flags.
  The old build script names forward to the unified build. No gas equations,
  gas timestep, chemistry, wall closure or particle operator is changed.
* CHMT keeps conserved gas on device throughout a coupling window. Only the
  final gas endpoint is downloaded for the material participant; commit avoids
  an upload immediately overwritten at the next window's import.
* Gather primitive/gradient/flux values for interface faces into one packed
  buffer; scatter only overwritten interface fluxes, and gather the limited
  accepted flux for the existing conservative packet ledger. These kernels only
  marshal data. Shared numerical kernels and packet accounting are unchanged.
  Topology is fixed as required by the existing backend; moving metrics are
  still refreshed through the existing geometry path.
* Skip redundant full mesh upload for static geometry, repeated zero-sweep
  material transport evaluation, and reconstruction for a zero ALE mesh flux.
  Physical Darcy, diffusion, reaction and nonzero moving-mesh paths remain active.

## Correctness evidence

Five small native cases (thermal, single-gas, diffusion, dry pyrolysis, moving
recession) have byte-identical complete physical checkpoint payloads and terminal
accepted/gas/material/exchange CSVs before and after optimization. Every measured
larger-grid CHMT repetition also matches its baseline byte-for-byte. All three
gasUGKP cases have identical final physical field files in every repetition;
the required physical end time is checked, not inferred from process exit.

The new sparse CUDA transfer kernel is executed with Ns=2 and Ns=3, 129 unsorted
interface faces across a partially occupied second thread block, noncontiguous
owner cells, mixed and single-gas modes (inactive species pointers are null).
Checks cover every primitive/gradient component, species layout, sparse writes,
and preservation of non-interface fluxes. Both native runs pass Compute Sanitizer
with zero errors. Full optimized dry CHMT also passes memcheck. Seven dry CPU
core/adapter regressions pass. Typed API tests exercise Ns=2/Ns=3 and build controls.

The additional finite-rate boundary-layer plate check reaches 1e-5 s over ten
coupling windows with identical complete physical state and CSVs. It is a short
execution-parity check only: pre-existing unbracketed surface thermal residual
warnings remain. The longer 1e-4 s baseline experiment was stopped after about
three minutes of slow progress; it is not claimed to have passed. This is not
evidence of long-time boundary-layer/surface convergence or calibration.

## Timing

| Application / case | Cells | Before (s) | After (s) | Speedup |
|---|---:|---:|---:|---:|
| gasUGKP / gas_species_wave | 128 | 2.4522 | 2.3760 | 1.032x |
| gasUGKP / small_sod | 200 | 0.7001 | 0.7108 | 0.985x |
| gasUGKP / flatplate_fixed | 512 | 1.1891 | 1.0388 | 1.145x |
| CHMT / thermal | 512 per region | 5.3149 | 5.1026 | 1.042x |
| CHMT / dry | 512 per region | 7.0347 | 6.9211 | 1.016x |
| CHMT / receding | 8 per region | 13.2842 | 11.9966 | 1.107x |

Medians of five fresh-case executions per variant after a discarded warmup pair;
baseline/optimized order alternates. Times include CUDA/process startup, solver
execution, diagnostics and output, but exclude case generation, meshing and
initialization. Single RTX4060, serial rank, identical before/after input and
compiler options. CUDA uses the explicitly recorded fast-compile native build,
not a default fully optimizing release build. No simultaneous GPU workloads were
launched during timing; ordinary OS scheduling noise is visible in raw samples.

The small Sod case is slightly slower in median (about 1.5%); there is no claim
that every case accelerates. The 512-cell-per-region thermal/pyrolysis gains are
small relative to timing spread. Moving recession and the gas plate show clearer
benefits. These measurements do not establish large-mesh, multi-GPU or MPI scaling.

## Observed transfers

Nsight Systems on the same 8-cell thermal case (100 gas microsteps):

| Operation | Before | After |
|---|---:|---:|
| H2D transfers | 6207 | 390 |
| D2H transfers | 6601 | 1817 |
| Total copied bytes, rounded MB | 3.020 | 0.885 |
| cudaMemcpy calls | 12708 | 1907 |
| CUDA kernel launches | 2400 | 2700 |
| cudaDeviceSynchronize calls | 400 | 400 |
| cudaStreamSynchronize calls | 200 | 400 |

Transfer count falls 82.8% and copied volume about 70.7%. Sparse gather/scatter adds
three kernels per microstep. Explicit stream waits increase for the packed host
consumers; this is **not** presented as a reduction in synchronization-call count.
Many blocking full-field copies disappear, but common Courant/diagnostic barriers
and CPU material work remain. Profiler timings are not used as benchmark timings.

## Test migration and pre-existing debt

Remove gasUGKP-only socket/wire tests for the retired transport. Keep the FSH wire
cases, migrate retained adapter fixtures to the typed namespace, and test direct
linkage/API signatures and build controls. The legacy gas suite has two pre-existing
collection errors (recursive source expansion). With those two files excluded,
PR24 gives 226 passed / 134 failed / 92 errors / 107 skipped; PR25 gives
217 passed / 123 failed / 92 errors / 107 skipped after the protocol-test migration.
There are no new failing identifiers among retained baseline tests. Counts are
not an overall test-suite success and are not directly comparable because obsolete
wire tests were removed. The CHMT historical orchestration fixture also still
uses obsolete terminal/rollback signatures from before PR24; its changed download
declaration is migrated but that fixture is not claimed as passed.

## Reproduce

Use `tool/gasUGKP/build_cuda_solver.sh` and `applications/CHMT/Allwmake` in separate
before/after work areas. Set `GAS_UGKP_BUILD`, `CHMT_BUILD` and `FOAM_USER_APPBIN` to
private output directories; use the same `NVCC`, species count and architecture.
The corrected PR24 baseline and toolchain flags are recorded in evidence.json.
After sourcing OpenFOAM and setting `CHMT_CUDA_STACK_BYTES=65536`:

```
python3 validation/resident_execution/benchmark_gas.py --before /path/pr24/gasUGKP --after /path/pr25/gasUGKP --source "$PWD" --cases "$UGKP_CASES_ROOT" --output /fresh/gas --repeat 5
python3 validation/resident_execution/benchmark_chmt.py --before /path/pr24/CHMT --after /path/pr25/CHMT --source "$PWD" --cases "$UGKP_CASES_ROOT" --output /fresh/chmt --repeat 5
nvcc -std=c++17 -O2 -arch=sm_89 -DUGKWP_GAS_SPECIES=2 -Iapplications/CHMT -Icommon validation/resident_execution/test_surface_transfer.cu -o /work/surface-ns2
compute-sanitizer --tool memcheck --error-exitcode 99 /work/surface-ns2
```

Repeat the last two commands with species=3. `native/run_native.py` from the PR24
review reproduces the five full solver cases and dry memcheck. The cases checkout
and input-generator hashes are recorded in PR24's evidence manifest. All measured
samples, physical-state hashes, profiler summaries and native memcheck evidence
are saved beside this report. A transient Python standard-library initialization
failure occurred during one case generation; it happened before solver launch,
was preserved, and a fresh retry supplied that sample. No timed solver sample was
discarded except the explicitly planned warmup pair.
