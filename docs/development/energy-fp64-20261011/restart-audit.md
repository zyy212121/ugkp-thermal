> Historical evidence note (updated 2026-10-11): blocked-migration and old
> coupling conclusions below describe the pinned earlier snapshot, not the current
> paired-carry implementation. The current design/cost contract is
> [paired-carry-design.md](paired-carry-design.md); current verification and
> remaining native-runtime limits are in [current-verification.md](current-verification.md).

# CHT authoritative-energy restart audit and guard plan

## Decision and supported contract

CHT persists `rhoE` as an OpenFOAM scalar energy-density field, not an integrated cell inventory. Its dimensions come from `p/(gamma-1) + rho*U^2/2` (`createFields.H:351-355`). Both resident frontends require DP OpenFOAM scalar storage and 32-bit labels (`GpuResidentStrict.H:2585-2593`), and their scalar-field pack/assign operations preserve those doubles. The C ABI already uses double pointers; after the owner promotion, the templated upload/download helpers select direct double copies. Existing restart values retain only the precision they already contain; an old float-valued field is promoted exactly, without reconstructing lost increments from primitive fields.

No thermal restart schema bump is needed for same-build compatibility: the wire representation was already the DP OpenFOAM field, with field dimensions and ordinary time-directory identity. Widening the resident authority changes the retained information, not the wire layout or units. The actual Foundation OpenFOAM 10 `src/OpenFOAM/db/IOobject/IOobjectWriteHeader.C:35-64` writes format, class, location, and object, but does not emit architecture/scalar-width metadata. Width is therefore guaranteed by the executable DP contract, not self-described in the field. No new mixed serialized format is introduced, and no additional mandatory width tag is required for the supported same-build restart. A future cross-build portable format with explicit scalar-width metadata is a separate change. Density representation and completed-step identity are already fixed by this owner and do not need a new stage schema. The thermal v4 manifest hashes `fluid/rhoE`, validates completed exchange flags and times, and includes pending wall-energy ledgers when the last exchange precedes the checkpoint. Persistence runs only after the completed solver step and rejects a pending thermal exchange. It does not persist an in-progress RK stage.

This conclusion applies to the actual CHT frontend: fixed `fvMesh`; constant perfect-gas thermodynamics (`readGpuGasConfiguration.H:81-115`); pure-gas, particle, and zero-particle gas/solid bridge routes; RK1/2/3 fixed-grid transport. Its resident owner has neither `gasSpecies` nor `gasGeometry`. Shared-kernel support for mixture, ALE, or another owner's accepted/trial snapshots does not add those capabilities to CHT. `rhoENext` is the RK initial state, not a rejected-step rollback checkpoint. There is no generic CHT gas rejection/retry checkpoint to migrate.

## Confirmed gap

The existing lossless-write guard is reached only through thermal restart preflight. Nonthermal pure gas writes at `diluteUgkwpFoam.C:549-576`; nonthermal particle mode writes at `:926-951`. Either can serialize `1e9 + 0.03` at low/default ASCII precision and lose the authoritative increment. A startup-only check is also insufficient if runtime-modifiable output settings are changed before a later write.

## Minimal implementation plan

- [x] Add focused tests first. Compile the production guard body with a small OpenFOAM dictionary/error adapter, testing missing/low ASCII precision, precision 16/17, and binary output. Assert both nonthermal write routes guard immediately before writing, and a common startup check precedes either initialization route.
- [x] Record RED evidence before implementation.
- [x] Add a local `requireLosslessGasEnergyRestartPrecision(const dictionary&)` in `diluteUgkwpFoam.C`. Require binary output or explicit ASCII precision at least `numeric_limits<scalar>::max_digits10`. Call before initialization and before either write route. Keep all existing thermal preflight checks intact.
- [x] Exercise exact ASCII-17 and binary scalar round trips and the old-float promotion boundary with production copy helpers and the actual CHT double authority type. This is a host-side lifecycle probe, not an OpenFOAM/CUDA runtime claim.
- [x] Run focused tests and relevant existing CHT restart contracts; report broader suite availability and failures. Commit only this frontend guard, its focused tests, and this audit document. No README, common, or GPU code edits.

## Additional audit findings

The current energy write-set is covered by divergence, RK save/blend, legacy repair, gravity, volume-fraction work, drag, and Eulerian particle heat. Particle relaxation/pressure/radiation paths write particle state, not another hidden gas-energy owner. Actual-particle publication tests subsequently found baseline local-kernel finite-step heat/drag residuals (not a complete-advance defect proof) and a new mixed-precision heat budget deficit: double gas energy retains a decrement that float particle temperature can round away. Full gas/particle pairing is not established and blocks production release of the precision migration; see particle-budget.md. Typed allocation/free and POD pointer refresh already follow each pointer's actual type (`GpuResidentStrict.cu:823-924`); no `sizeof(GpuReal)` gas-energy copies were found.

## Evidence

- RED, before the production edit: the focused file reported 2 failures because the precision guard and startup/write-route calls were absent, plus 1 passing double-copy/serialization/old-float boundary probe.
- GREEN: `PYTHONPATH=/workspace/scratch/98b6339ec81b/chmt_implementation_20261011/python_deps PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider applications/CHT/devtools/tests/test_authoritative_energy_restart.py` passed all 3 tests.
- Adjacent pending-ledger and pure-gas contracts: 9 passed, 1 preexisting failure. `test_pending_wall_energy_restart_contract.py::test_gpu_transfer_is_compact_and_limited_to_coupled_patch_ranges` expects the token `ugkwpGpuResidentStrictPeekWallEnergyLedgerRange` inline in `GpuResidentStrict.cu`; that token is also absent from base `ab900f54`, which uses the shared implementation.
- Native OpenFOAM 10 frontend syntax check passed: `g++ -std=c++17 -fsyntax-only -DUGKWP_USE_CUDA -DWM_DP -DWM_LABEL_SIZE=32 -DNoRepository`, with CHT/common and OpenFOAM finiteVolume, meshTools, OpenFOAM, and POSIX include paths. No object or binary was written.
- The parent owns overall verification. Whole-repository pytest collection has 38 reported preexisting duplicate-basename collection errors and requires separate test-root invocations. This worker does not claim a full passing suite, GPU runtime restart, or cross-build binary field compatibility.
- `git diff --check` passed for the focused patch.
