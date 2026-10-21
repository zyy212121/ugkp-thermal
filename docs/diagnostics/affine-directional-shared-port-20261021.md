# Qualified affine species / directional FD shared port — 2026-10-21

## Status and scope

This delta follows local integration `2ef33658070b1f1adf950ad55dac22afa2796919`.
The **CUDA source is prepared, but nvcc, GPU execution and CUDA sanitizers are
UNRUN** here. The user runs device validation. No CPU parallel/cooperative test
or acceleration measurement was performed for this delta. The earlier
[integration report](coherent-cpu-shared-integration-20261021.md) records historical
work, not a fresh qualification of this delta.

The separately preserved mathematical candidate completed the mixed no-blowing
200 microsecond trajectory: 672,000/672,000 roots, unchanged original timestep,
`maxIterations=60` and tolerance `1.01e-8`, 179 source pins unchanged. This is
observed qualification, not a universal nonlinear convergence guarantee. It does not establish exact canonical
wall-function/transient entrance agreement (the separate math audit records the
remaining few-percent/entrance differences). The new species representation covers frozen binary, equal positive diffusivities,
zero species wall mass flux, regular interior moments, and exact uniform pure
species. General pure-top/nonuniform extension is not claimed solved; unselected
inputs continue through the unchanged legacy fitting routine.

## Exact mapping and unchanged semantics

- `ConservativeProfile.H`: free wall composition with Neumann species flux,
  two moment-fitted global cubic-logit coefficients in a fixed piecewise-affine
  coordinate, and exact uniform pure shortcuts. Actual interface publication
  uses the same common diffusion flux `J` for both species, with `sum(h*deltaJ)`
  applied to energy. It does not merely average primitive gradients.
- `WallCoherentNumerics.H` and `WallCoherentSolver.H`: mirrored colored FD before
  shrinking, and individual directional recovery when colored response is zero
  or unresolved. Actual representable steps form denominators. Signed
  `a=sqrt(k)` coordinates remain allowed (`k=a*a`). Alternative-probe checks do
  not restrict the original ordinary/clipped-Y scalar direction or physical input.
- Physical residual equations, LU policy, line search, physical tolerances,
  existing input/capacity behavior and publication/error semantics are unchanged.
  Affine helper/finite flags, determinant, fit residual and iteration exhaustion
  are observations, not new hard rejection conditions. `fitTeam` still publishes
  its last selected affine fit. The legacy incompatible zero-inventory case
  retains its prior false return.
- Native warp, pool and split architecture is unchanged. LU/warm state remains
  wall-owned; accepted history is separate. Opposite-probe saved steps belong to
  worker scratch, not each wall/lane. Affine coefficients are lane-local but
  team-uniform; quadrature storage is worker-owned and interface mobilities are
  broadcast before use. Existing trace/source publication consumes the fitted
  profile immediately; no new persistent trace state is introduced.

The existing production `128*8` samples cover the two contiguous segments:
8 Gauss samples per intersected microinterval, with at most one additional split
interval. The prototype's diagnostic `1024*8` extent was **not** copied and no
new capacity check was added. Four extra doubles/sample add 32 KiB per profile
worker. FD scratch adds `48*M` bytes (1,152 / 2,304 / 6,144 for M=24/48/128).
Measured host solver scratch sizes are 216,120 / 760,536 / 4,971,416 bytes;
wall-state/LU sizes remain 66,528 / 129,504 / 339,424. Device resource cost is not
measured here.

## Scalar evidence for this delta

All commands ran sequentially after the exclusive serial timing window ended.

- Red-before-port checks reproduced the anchored wall composition and failed
  fresh-cache 723 root. New self-contained regression suite: **3 tests passed**.
- Manufactured mixed/pure profiles and 24/48/128 capacity cases: maximum moment
  error `1.28e-15`; wall composition moves from 0.8 to 0.820017218037892 in the
  mixed regression. Pure species are exact, wall species flux is zero, and the
  nonlinear unequal-mobility interface publishes common J plus its enthalpy.
- A test-only zero-iteration fit returns true with residual 0.0183333333333334:
  the diagnostic does not become a publication veto. No production budget changed.
- Numeric captured-723 fixture: accepted-state, failed-terminal-state and cold
  fresh-cache solves match pinned y/state/residual arrays, fluxes, and counts
  exactly. Failed-state repair: 5 iterations, 208 F evaluations, 5 factorizations,
  residual `9.347169785869279e-9`. An isolated temporary header copy disables only
  the batch mirror to exercise the actual scalar recovery path: 7 iterations,
  385 F, 6 factorizations, residual `6.058429544762585e-13`, also exact. No test
  switch is added to production. These do not reconstruct unavailable pre-call
  warm-cache chronology.
- Independent frozen-candidate profile parity: owners 90 and 7782, each at
  capacities 24/48/128, all **6 dumps byte-identical and finite** for fit, values,
  derivatives, physical fluxes, sources and energy terms. Higher capacities
  resample the captured transformed state, not converged fine-grid roots.
- Corrected CUDA driver's CPU-only paths: 11 profile cases plus 3 coherent solver
  modes passed. Optional unavailable chemistry mismatch remains a documented NaN
  placeholder; other compared physical output fields are finite. Comparator
  semantics are exercised by the CPU-only build too.
- Existing scalar numerics (2 tests) and the single scalar wall-storage ownership
  test passed. The cooperative test in that file was explicitly not run.
- Relocated detached checkout with only tracked base plus these 12 intended files
  passed the 3 scalar tests and both driver CPU-only paths; the nine excluded local
  files were absent. Profile address/undefined sanitizers passed at 24/48/128 with
  `ASAN_OPTIONS=detect_leaks=0`. LeakSanitizer itself could not run under the
  environment's ptrace restriction; its failed launch is retained in the logs.
- Independent source review found no pinned-math, status, ownership or collective
  uniformity blocker. This is not substitute evidence for actual CUDA execution.

The fixture and expected roots are numeric source, with no raw pointer-bearing
snapshot dependency. See
[`coherent_directional_723.provenance.json`](../../validation/tests/gas_wall/fixtures/coherent_directional_723.provenance.json)
for capture and frozen-output hashes.

## Reproduce from the checkout

Scalar only:

```sh
python3 validation/tests/gas_wall/test_affine_directional_scalar.py
python3 validation/tests/gas_wall/test_coherent_shared_numerics.py
python3 validation/tests/gas_wall/test_coherent_wall_storage.py Storage.test_worker_scratch_and_wall_history_are_distinct
python3 validation/cuda/wall_coupling/reproduce.py --cpu-only --output /tmp/coherent-scalar
```

User-run CUDA (select the GPU's architecture):

```sh
python3 validation/cuda/wall_coupling/reproduce.py \
  --cuda /usr/local/cuda --arch sm_89 --sanitize --output /tmp/coherent-cuda
```

`profile.cu` uses current shared `fitTeam(..., SerialWallTeam{})` rather than the
old `fit()` or an external stale header. Its original eight cases retain legacy
BVP-generated seed states; they do not qualify the coherent solver. The separate
`coherent_directional.cu` invokes actual `Engine<24, CudaWallTeam>` for all three
723 modes and compares current scalar results, original residual tolerance,
state and published output. Counts are recorded, not required to match across
architectures. Both complete device computations stay on the GPU; CPU runs are
validation oracles only. Existing legacy CUDA BVP tests remain separately named.
The runner records source/binary hashes, commands, logs and explicit execution
status. No speedup or GPU pass is claimed before user execution.


后续本机原生 CUDA、整算例、守恒及效率验收见 [2026-10-22 报告](pr22-native-cuda-20261022/RESULTS.md)。本文的 UNRUN 状态保留为当时记录，最终结论以新报告及原始证据为准。
