# PR14 FAST F002/F004 GPU recheck

This fixture includes each complete production CUDA translation unit. It uses
actual CUDA allocations, H2D/D2H copies, native configuration/patch upload, and
both production device initializers. It contains no CPU CUDA shim. Compilation
is not evidence that the assertions have executed.

## Focused real-GPU regression

From the repository root, on the RTX 4060 review host with its CUDA toolkit:

```
python3 tests/fixtures/pr14_fast_cold/run.py . /tmp/pr14-cold-CHT CHT
python3 tests/fixtures/pr14_fast_cold/run.py . /tmp/pr14-cold-FSH FSH
```

Override `--nvcc /path/to/nvcc` and `--arch sm_XX` for the actual toolkit/device.
Use `--compile-only` to compile without claiming runtime results. The runner
records the exact command, build log, and (only when executed) runtime log.
CHT uses O3/sm89/fmad=false; FSH retains its native fmad=true setting.
`PR14_FAST_GPU_TESTS=1 pytest -q tests/test_pr14_fast_cold_initialization_cuda.py`
runs the same native regression. Without the opt-in, these tests are skipped.

Coverage:
- Configure-first and patch-upload-first owner creation
- Uncoupled faces seeded with sqrt(1800*710*100), the existing CHT scalar source
- Selected coupled patch values retained across repeated configure, heat disable,
  re-enable, and subsequent patch replacement; canonical/published pointer match
- No newly imposed restriction on supplied zero or signed patch values
- Both initializers preserve uniform profile enthalpy plus existing double carry
  through the CHT float cache cast, including zero/positive/negative prior carry
- FSH's unchanged double profile path publishes initial enthalpy exactly

## Repeat the original native failure cases

Use unchanged review input directories:

- review/pr14/reviewer-deliveries/20261013T182722Z-fast/inputs/native/resident/CHT-coldWall1D-INPUT2
- review/pr14/reviewer-deliveries/20261013T182722Z-fast/inputs/native/resident/CHT-coldWall2D-INPUT2

Copy each into a fresh scratch case, build the formal current CHT source with
applications/CHT/tools/build_cuda_solver.sh, run blockMesh and CHT, and retain
commands, source/binary hashes, build/run logs, and every written restart/field.
Do not overwrite the original or diagnostic review evidence. Re-run under
Compute Sanitizer memcheck and synccheck. Check every written timestamp,
especially first cold contact, for finite scalar/vector/profile/carry state and
mean(profile converted to float64) + double carry caloric temperature mismatch
<= 1e-9 K, using the independent equations/check in
inputs/resident_check_fp64_mean.py from that review delivery. Its artifact paths
are delivery-specific; point the checker at the new outputs rather than reusing
old diagnostic results. Preserve the original thresholds and input physics.

Repeat the recorded split/restart sequence and compare all terminal particle
members and fields, including float profiles and double carry/history. The
review's P007/P010 runs are historical diagnosis, not a pass of this formal fix.

Mixed coupled/uncoupled patch behavior in a real OpenFOAM case, actual first
contact, restart, and full energy budgets remain application-level review work;
the focused initializer fixture is not a replacement. No speedup is claimed.
The producer adds one initialization-only host fill/H2D copy, replacing one
zero fill. Completed owners do not allocate, refill, or recopy at every step.
