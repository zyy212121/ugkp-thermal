# PR22 native CUDA qualification reproduction

Run from the repository root on Linux/WSL with OpenFOAM 10, Python 3 + NumPy,
g++ C++17, CUDA 13.1 and an sm_89 GPU. The qualified input uses **24 nodes**,
zero wall mass/species injection, mixed Air/CO2 and nonzero reconstructed
interior species transport. No physical tolerance, timestep or Newton budget
is relaxed. See RESULTS.md for measured scope and remaining physical limits.

## Frozen independent CPU reference

Follow `../pr22-math-qualification-20261021/REPRODUCE.md` to join/extract the
three committed reference parts. Check the combined archive and run its
`verify.py` **before** rebuilding any shipped executable. The original package
contains 748 manifest entries. Set `REF` to the extracted
`pr22-math-reference-20261021T0941Z` directory and `OUT` to a new work directory.

On Ubuntu 22.04, the shipped CPU executable requires a newer glibc/libstdc++.
Rebuild the unchanged archived source with its `directional-candidate/build-coupled.sh`
from the extracted reference root. Run `bash run-coupled.sh 20 "$OUT/cpu-4us"`. Also run
`bash run-coupled.sh 1000 "$OUT/cpu-200us-local"` to quantify local-toolchain
CPU drift against the preserved final fields; do not silently replace the frozen reference.
The preserved 1000-step CPU fields are under
`source-sign-audit/math-gap-20261021T0626Z/directional-candidate/TY-directional-200us`.

## Native operators, chronological replay and sanitizers

```sh
python3 validation/cuda/wall_coupling/reproduce.py \
  --cuda /usr/local/cuda --arch sm_89 --sanitize \
  --math-reference "$REF" --output "$OUT/native"
```

This runs 11 profiles, three captured-723 solver modes, six legacy cases,
and the 4000 consecutive solve/profile/source records for owners 0 and 7782
against **independent frozen final-A.csv**. The current scalar implementation
is a separate device-parity oracle. All three compute-sanitizer tools run on
each operator family; chronological sanitizer runs use four consecutive calls
per owner. Scope/status, commands, source/binary hashes and raw outputs are saved.
SequenceRecord pointer bytes are rehydrated, never dereferenced as saved pointers;
its 1216-byte little-endian x86-64 ABI is checked. The reconstructed interface
species flux is measured after subtracting advective transport; near-zero BVP
matching diffusion alone is not evidence of active species transport.

## Full production application

The production script now enables --expt-relaxed-constexpr, matching the native
operator driver; omitting it caused all 336 coherent walls to fail, and adding
only this flag fixed the Euler control. The global GasConservativeMatching include fixes first inclusion inside an
anonymous operator namespace. On this 16 GiB Windows laptop, stock full-backend
ptxas exhausted the original WSL memory allowance. The measured build retains
nvcc -O3, FP64 and fmad=false, with this explicit CUDA fast-compile option (trades some device optimization for compile time):

```sh
source /opt/openfoam10/etc/bashrc
ulimit -s 65536
export FOAM_USER_APPBIN="$OUT/runtime/bin"
mkdir -p "$FOAM_USER_APPBIN"
export NVCC_APPEND_FLAGS='-Ofc=mid -v -Xptxas=-v'
export GAS_UGKP_FMAD=0
export GAS_UGKP_CUDA_STACK_BYTES=131072
/usr/bin/time -v bash tool/gasUGKP/build_private_backend.sh
```

The intermediate assembler-only --allow-expensive-optimizations=false attempt
was manually stopped after 33 minutes; -Ofc=max changes the complete device
compilation pipeline and completed the backend. The final qualified build uses
-Ofc=mid for better runtime efficiency; the intermediate max build and interrupted
20-step timing attempt are explicitly distinguished in the logs. See logs for outcome, memory and duration; this option is not a
claim that the stock full-backend build passed. Fast compilation outlines device calls; an explicit 128 KiB thread stack is
needed for the observed thermodynamic call chain. The unset runtime environment
retains the CUDA default; the configured limit is printed and recorded. The
no-stack run and its memcheck stack-overflow report are retained.
Local temporary swap files used
for compiler memory are documented and removed before measured application runs.
Do not modify permanent WSL configuration merely to reproduce this report.

Prepare cases with `prepare_math_case.py`, **not only the original pure-gas
restart template**. The helper restores the frozen mixed initialized state and
boundary composition, fixes nodes=24, startFrom=startTime, adjustTimeStep=false,
writeControl=timeStep, and records input hashes. The comparator checks actual
step count, every dt/time and cumulative wall solve count; a matching terminal
time alone is insufficient.

```sh
python3 validation/cuda/wall_coupling/prepare_math_case.py \
  --reference "$REF" --output "$OUT/gpu-4us" --steps 20
python3 validation/cuda/wall_coupling/run_case.py \
  --case "$OUT/gpu-4us" --runtime "$OUT/runtime" --output "$OUT/gpu-4us-audit"
python3 validation/cuda/wall_coupling/compare_math_case.py \
  --case "$OUT/gpu-4us" --cpu "$OUT/cpu-4us" \
  --audit "$OUT/gpu-4us-audit" --output "$OUT/gpu-4us-comparison.json"
```

Repeat preparation with `--steps 1000` for 200 microseconds. Compare against the
frozen `TY-directional-200us` directory and add
`--wall-function "$CANDIDATE/all-wall-affine-comparison.csv" --step 1000`, where
`CANDIDATE="$REF/source-sign-audit/math-gap-20261021T0626Z/directional-candidate"`.
The wall mass flux must be exactly zero. Individual species face-flux roundoff
is checked per face against 64*FP64 epsilon times the maximum sampled interior
species face flux in kg/s; the two wall species must cancel exactly. This is
a recorded roundoff-scale acceptance rule, not a changed physical solver
tolerance. The independent BVP output wallJ remains exactly zero.
The additional macro parity criterion is `abs(GPU-CPU)/max(1,abs(CPU)) <= 1e-5`.
It is separate from the original nonlinear residual criterion. A comparison
exit 1 must be reported even when every nonlinear solve converges.
All 336 walls are compared without tuning or exclusions. Physical wall heat/shear
come from **wall-work.csv diffusiveEnergy / tractionX**, after reconstruction;
BVP heatFluxIntoGas is a different quantity. The activity-normalized heat error
is explicitly not a relative error of net heat flow.

For an Euler conservation control, use:

```sh
python3 validation/cuda/wall_coupling/prepare_math_case.py \
  --reference "$REF" --output "$OUT/gpu-euler" --steps 1 \
  --integrator Euler --audit-input "$OUT/mixed-euler-input"
python3 validation/cuda/wall_coupling/run_case.py \
  --case "$OUT/gpu-euler" --runtime "$OUT/runtime" --output "$OUT/euler-audit"
python3 validation/cuda/wall_coupling/audit_euler.py \
  --case "$OUT/gpu-euler" --input "$OUT/mixed-euler-input" \
  --audit "$OUT/euler-audit" --dt 2e-7 --output "$OUT/euler-budget.json"
```

The exported raw mesh/mixed fields feed the independent inventory audit; the
actual GPU face-flux and source arrays are saved by the application audit. The report records the excluded algebraic owner
omega constraint and a 1e-12 normalized inventory-defect criterion. Repeat a
fresh Euler case with `run_case.py --sanitize memcheck`, `racecheck` and
`synccheck`; `--target-processes all` covers the private backend child.

## Performance and evidence integrity

Run GPU timings sequentially without other GPU work. Set `UGKP_PROFILE_BENCH=1`
when executing the built native `profile` binary to measure 336 independent
reconstructions using CUDA events, single lane versus warp32. This is a
profile-only fixed-input GPU comparison, not a coupled-step or CPU speedup.
The chronological one-wall replay includes launches and managed-memory overhead
and must not be interpreted as throughput acceleration.

The full-step CPU reference includes passive failure-capture copies and extensive
CSV output; the GPU application has different I/O. Any elapsed-time ratio of
those runs is an end-to-end observation with those scopes, not equal-cost
operator timing. Original timing/event/telemetry logs accompany the results.

Evidence archive parts, SHA256SUMS and a per-file manifest are stored beside
this document. Join parts in filename order, check SHA256SUMS, then extract.

```sh
cd docs/diagnostics/pr22-native-cuda-20261022
cat evidence.tar.gz.part[0-9][0-9] > pr22-native-cuda-evidence-20261022.tar.gz
sha256sum -c SHA256SUMS
tar -xzf pr22-native-cuda-evidence-20261022.tar.gz
```

The archive excludes executables, swap, generated PTX and duplicate CPU reference
archives; those are reproducible from tracked sources and the pinned reference.
Earlier failed build/test attempts are retained with their failure scope.
