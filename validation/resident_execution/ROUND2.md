# PR25 follow-up: owner streams, completion and host execution

This round starts from the user's path-corrected PR25 commit
`63158141dd82c88d60e848463271a1eeea64da08`. The baseline is this PR25 state,
not PR24. gasUGKP remains the pure-flow application, CHT the conjugate-heat
application, and CHMT the conjugate heat/mass-transfer and ablation application.
The changes apply to their existing execution paths; there is no case-specific
operator, substitute application, or FSH-specific implementation change.

## Changes and ownership

* CHMT owns a nonblocking CUDA stream. Initialization, mesh/state uploads,
  zeroing, numerical kernels, status checks, interface transfers and diagnostics
  use that stream. Configuration candidates inherit it. Buffer replacement,
  rollback and destruction complete the owning stream before releasing storage;
  they do not rely on the legacy default stream to establish dependencies.
* A reusable pinned transfer queue retains each H2D source until completion, so
  temporary host vectors can leave scope after submission. D2H copies first land
  in queue-owned staging. Sparse wall/interface results and the diagnostic
  context complete together; the host consumes numerical candidates only after
  successful completion and fatal-diagnostic observation. The complete gas
  endpoint, including SST fields when enabled, uses one batched completion.
  Failed copies and C++ allocation exceptions drain pending DMA and discard
  unpublished destinations. A later successful drain permits recovery from a
  recoverable failure; persistent CUDA errors remain failures.
* Shared gas stage checks scan raw status on device and retain the first
  `InvalidStorage` location in deterministic cell-before-face/index order.
  Existing producer diagnostics take precedence. Stage checks perform no host
  readback or wait. gasUGKP observes completion at its public interval/Courant
  boundaries; CHMT observes before each host wall/ledger consumer and before
  publishing an accepted microstep. Existing numerical kernels and private gas
  callbacks now consistently stop at their entry when a sticky fatal is present.
  These guards change failure execution, not valid-state arithmetic.
* The diagnostics owner accepts an already completed snapshot and explicit-stream
  initialization/observation while retaining its existing no-argument API. CUDA
  transfer failures share one catalogue producer. The CHMT unavailable-storage
  fallback has its own source ID, preserving the generic transport producer's
  location and meaning.
* Completed gas history is borrowed as an immutable reference until the next
  window begins. Material consumers borrow the selected record's gas fields and
  use cursors instead of repeatedly finding the held record from the beginning.
  Backward queries reset the cursor. The dry surface evaluation borrows its
  current state because it does not consume the film edge-sweep field previously
  cleared in a copied state. Source-variation comparisons retain their order and
  arithmetic. CHMT geometry preparation also borrows `trial`: CFL and RK callbacks
  do not modify host geometry before the one endpoint publication, so the full
  `microBase` snapshot and its per-microstep copy are unnecessary.
* Geometry reuse requires identical coordinate representations, including signed
  zero. Original midpoint/third-point interpolation arithmetic is retained; a
  sampled geometry is reused only if its actual coordinates match exactly.
  Wall-distance evaluation parallelizes independent cells with OpenMP while
  preserving each cell's face/triangle traversal and minimum order. Sufficient
  work is required to amortize startup, and nested parallel callers retain their
  own execution. Floating-point environment and error handling are preserved;
  unsuccessful parallel evaluations are replayed in the original serial order.

Gas equations, fluxes, timestep selection, chemistry, wall constitutive models,
material packet accounting and RK arithmetic are unchanged. The remaining
host wall/ledger and Courant dependencies are real completion boundaries; this
change does not claim that the whole coupled solve can run without waits.

## Correctness evidence

The final combined targeted selection passes **35 tests**, with the two previously
identified wall-fixture tests deselected. Its exact test list and options are in
`round2-evidence.json`; the earlier focused selections below overlap with it.

* The CHMT transfer/diagnostics/reconfiguration selection passes **18 tests**.
  Delayed host CUDA shims exercise temporary-source lifetime, separate-owner
  streams, staging reuse, publication after completion, failed copies, failed
  completion, allocation exceptions, configuration rollback, topology rebinding
  and destruction. These are host ownership tests, not GPU execution claims.
* The focused gas execution selection passes **13 tests**, covering completion,
  status ownership and the updated production interfaces.
* Geometry is compared against the retained serial PR25 oracle in **128 exact
  comparisons per execution**, with OpenMP disabled and with one/four OpenMP
  threads. The checks also pass with `-O3`. They cover both successful output and
  failure behavior, including stationary and moving geometry, floating-point
  environment and nontrivial wall-distance workloads.
* The final native OpenFOAM10/CUDA CHMT binary runs all five small scenarios:
  thermal, single-gas, diffusion, dry pyrolysis and moving recession. Full dry and
  receding runs pass Compute Sanitizer memcheck with zero errors.
* The gas completion fixture includes the actual production backend and captures
  its requested-interval pipeline in a CUDA graph. It checks deterministic status
  selection, sticky producer priority, operation metadata, partial thread blocks,
  zero extents, fatal-gated kernels with unavailable numerical backing, preserved
  wall-history sentinels, and refusal to publish a stale Courant candidate.
  Ns=2 and Ns=3 native executions pass both memcheck and synccheck with zero errors.

All five final native cases have byte-identical complete physical checkpoint
payloads and terminal accepted/gas/material/exchange CSVs against the baseline.
The dry and receding sanitizer runs match those payloads too. The final short
finite-rate wall run (10 windows to 1e-5 s) also matches its baseline exactly;
the owner-stream version separately passed wall memcheck. Every measured CHMT
repetition has the same physical payload; all final gasUGKP physical field files
match in every repetition and reach the specified physical end time.
Independent small-case thermal/diffusion checks and total inventory checks pass.
The largest relative mass drift is zero and the largest relative energy drift
in the five cases is 3.96e-16. These checks do not waive existing accuracy warnings.
Three CPU gas adapters additionally reproduce all 324 original output lines
byte-for-byte. Raw measurements, hashes and toolchain details are recorded in
[round2-evidence.json](round2-evidence.json).

## Timing and profiler results

| Application / case | Cells | Before (s) | After (s) | Speedup |
|---|---:|---:|---:|---:|
| gasUGKP / gas_species_wave | 128 | 2.6216 | 2.6024 | 1.007x |
| gasUGKP / small_sod | 200 | 0.9047 | 0.9100 | 0.994x |
| gasUGKP / flatplate_fixed | 512 | 1.2125 | 1.2038 | 1.007x |
| CHMT / thermal | 512 per region | 6.2192 | 1.4872 | 4.182x |
| CHMT / dry | 512 per region | 8.0168 | 3.5412 | 2.264x |
| CHMT / receding | 8 per region | 13.4123 | 10.2451 | 1.309x |

Medians of five alternating before/after executions following a discarded warmup
pair. Times include process/CUDA initialization, numerical work, diagnostics and
output; case generation and meshing are excluded. No other agent build or GPU
experiment overlapped the measured pairs. CHMT uses 32 OpenMP threads on the
32-logical-CPU WSL guest. The original baseline binary is unaffected by that
environment setting. Gas improvements are small and overlap run-to-run noise;
the stable conclusion is reduced transfer/synchronization demand, not a large
small-case gas speedup. CHMT shows substantial benefits in these measured cases.

The before binaries were built at `81436ca`; the subsequent `6315814` changed only
the build-path helper, its test and README, with no runtime implementation change.
The final binaries are native sm_89 builds. Do not compare these wall times directly
with older sessions under different machine load.

| Trace | Device sync calls, before → after | Stream sync calls, before → after | D2H bytes, before → after |
|---|---:|---:|---:|
| CHMT thermal / dry | 400 → 0 | 400 → 612 | 6,415,440 → 2,680,744 |
| gasUGKP species wave | 1515 → 0 | 606 → 1212 | 5,486,036 → 367,760 |

These are full-run Nsight API counts and transfer bytes, measured separately from
wall timings. Both CHMT traces give identical counts; after-change kernels use the
owned nondefault stream. The increased stream-wait count replaces stage-wide
device waits and accompanies explicit host consumers; total explicit waits still
decrease. Allocation/free and CUDA-internal dependencies are not inferred away
from these API counts. No claim of zero synchronization or multi-GPU scaling follows.

A supplementary thermal thread sweep gave median times of 3.039 / 1.789 / 1.505 /
1.451 s for 1 / 4 / 8 / 32 threads, with exact output parity. It used two measured
repetitions per size after warmup, before the last unused microstep-copy removal;
it supports the scheduling choice and is not a scalability benchmark.


## Limits and pre-existing test debt

The existing wall waveform and unbracketed surface-thermal residual warnings
remain visible. This round does not establish long-time wall convergence,
constitutive calibration or a new numerical-accuracy result. The short native
wall checks retain their execution-validation scope.

Some historical tests still contain stale API/namespace or wall-matching fixture
expectations, and a storage fixture fails its existing `-Werror` build in unrelated
wall reconstruction headers. The wall expectation and build failures were
reproduced with the baseline storage implementation. Historical diagnostic
catalogue issues also remain outside this execution change. The old git-history
fixture bundle requires a prerequisite commit absent from this checkout; scoped
tests use `--noconftest` to avoid that unrelated session-start import. The passing
selections above are not an all-suite-green claim, and excluded legacy tests are
not counted as passes.

Measurements use one RTX4060 and the recorded OpenFOAM10/CUDA toolchain. They do
not establish MPI, multi-GPU or large-mesh scaling. Native CUDA compilation uses
`--Ofast-compile=max -Xcompiler=-fno-tree-fre`, with the recorded CHMT `-O2` and
gasUGKP `-O3 --fmad=true` settings and a 65536-byte device stack. These are explicit
fast-compilation builds, not a benchmark of default fully optimizing CUDA builds.
Host geometry's separate `-O3` oracle check does not alter that qualification.

## Reproduce the focused checks

Use the isolated before/after build layout and native build entry points in
[REVIEW.md](REVIEW.md), selecting `6315814` as this round's baseline. Keep species
shape, architecture, compiler flags, case inputs, thread count and runtime stack
settings identical for each measured pair. After activating the toolchain, run
from the repository root:

```sh
python3 -m pytest --noconftest -q \
  validation/solvers/CHMT/tests/test_host_transfer_queue.py \
  validation/tests/test_gpu_diagnostic_owner.py \
  validation/solvers/CHMT/tests/test_shared_reconfiguration.py -k 'not wall'
python3 -m pytest --noconftest -q \
  validation/solvers/CHMT/tests/test_geometry_execution.py
```

The geometry fixture builds both serial and OpenMP variants and runs the latter
with `OMP_NUM_THREADS=1` and `4`. Repeat its recorded compiler invocation with
`-O3` for the additional optimization-level check. Build
`test_gas_async_completion.cu` with the production gas backend's include paths and
flags for each species count, for example:

```sh
nvcc -std=c++17 --expt-relaxed-constexpr --Ofast-compile=max \
  -Xcompiler=-fno-tree-fre -O2 -arch=sm_89 -DUGKWP_GAS_SPECIES=2 \
  -Iapplications/gasUGKP/gpu -Icommon \
  validation/resident_execution/test_gas_async_completion.cu \
  -o /path/gas-async-ns2
```

Then run:

```sh
compute-sanitizer --tool memcheck --error-exitcode 99 /path/gas-async-ns2
compute-sanitizer --tool synccheck --error-exitcode 99 /path/gas-async-ns2
```

Repeat for Ns=3. The native five-case harness and `benchmark_chmt.py` /
`benchmark_gas.py` use the same case setup described in the first review; final
raw measurements, binary identities and physical hashes belong to this round's
evidence rather than the earlier PR24 comparison.
