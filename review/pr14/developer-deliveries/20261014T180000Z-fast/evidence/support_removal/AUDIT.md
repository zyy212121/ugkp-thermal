# SolidFilm support accounting removal, bounded implementation

Base: 0aa34bc1b0e370ec99e6107faf83ff7f0ac3e668. Working repository: pr14_fast_fix_20261014. No commit, push, README edit or new rejection boundary. Initial four dirty film/chemistry files were not edited by this worker. Later unrelated workers' edits are excluded from this patch.

## Data-flow and deletion boundary

- `CpuSurfaceInterface.C::evaluateCpuSurfaceImpl`: removed the local gradient -> filmProfile -> bottomShear -> phase.momentum chain. Every local result in this chain fed only `phase.momentum`. Kept actual surface constitutive solve, all phase mass/species, energy decomposition, liquid kinetic advection, wall traces and front velocities.
- `InterfaceMath.H::assembleSurfacePackets`: removed the analogous local G -> filmProfile -> bottomShear -> phase.momentum chain. Kept real primary gas momentum and pore mass-carried momentum; kept phase disappearance energy remainder.
- Neither SolidQ nor FilmQ contains an evolved momentum variable. `packetDelta` changes solid energy/masses and film enthalpy/masses, never solid/film velocity. `CpuMaterialDriver::filmForcing` consumes only phase mass/species/energy/kinetic channels. Momentum of these SolidFilm packets had only exchange/support ledger consumers, validation and serialization.
- Removed SurfacePacketIdentity.filmPressureGradient and its CpuSurfaceInterface parameter/validation/assignment. Removed material wrapper parameter and all five relevant call argument sites. CpuMaterialDriver no longer allocates, requests or copies the unused transport output vector. The CPU film transport's INTERNAL pressure-gradient calculation remains, feeding its actual profile/transport/energy. Its public optional output remains for other legitimate consumers/tests.
- `Budget.supportImpulse` had no physical consumer in production. Removed all accumulation in material addBudget/accountCpuPacket and Coordinator.accountAssembledPacket; removed its finite checks. It remains only a legacy checkpoint slot, including arbitrary old finite/nonfinite bit classes, not a runtime source or admission condition.
- `supportWork` remains physical reduced-film energy. Actual shear/velocity/kinetic profile in FilmMath, FilmContracts and CpuFilmDriver, EOS pressure, p/rho, phase pressure work, gas actual flux, reaction/pore/particle momentum remain untouched.
- ExchangeMomentum still exists at this bounded implementation snapshot for independent reciprocal actual-packet diagnostics. The later expanded task to remove other pure audits is owned by other workers and may supersede it.

## Serialization and semantics

Checkpoint.C is byte-identical to base (SHA256 6b67d22c84acfe6af68cca538d966d91ed0ec0fe49e05edaa8c3745e9e698739). Field order/schema/version are unchanged. Support slot is stored/read but never added or checked. SolidFilm uses the existing generic packet momentum slot initialized to zero; cannot remove that generic slot because actual gas/particle packets use it. New snapshots intentionally differ in diagnostic SolidFilm momentum and support accounting; no claim of full-state byte equality to old outputs.

Old nonzero support history can survive a resumed checkpoint but is stale legacy data. It must not be interpreted as a current support integral. No migration that invents a reaction or reinterprets old values was added.

## Tests and exact scope

- New `test_chmt_support_accounting_removal.py`: 1 PASS, 32.56 s. Compiles actual base and current producers and film/material/geometry consumers with FP64 and -ffp-contract=off. Logs hexadecimal FP64 physical fields and compares exact text. Both existing wet zero/positive-phase fixtures retain mass/energy, thermodynamic outputs, actual updated solid/film states, velocities and moved mesh points. Actual shared packet assembly also covers negative/zero/positive phase and nonzero shear/base velocity/gravity. Removal requirement fails on old source and passes on current source. NaN/+Inf/-Inf support slots no longer influence validBudget or packet accounting. Real GasSolid reciprocal impulse is unchanged.
- This is host-component production C++ coverage, NOT native OpenFOAM/CUDA coupled-run evidence. CPU signed phase packet assembly is not an end-to-end freezing trajectory.
- `check_cpu_film.sh`: PASS (planar and curved reduced-film energy, dry birth, evaporation and terminal dryout). Existing dirty evaporation fix was retained and thus part of current-tree run.
- Incremental pytest: 28 PASS, 3 SKIP, 2 FAIL, 1 deselected. Three skips are missing OpenFOAM. Failures: unavailable historical 360bf94 object; backend mechanical-pressure fixture's old Thrust Policy stub lacked allocator call. Neither is declared PASS. Other worker owns backend stub.
- `check_cpu_geometry_state.sh` old checkpoint assertion failed equally on exact HEAD and current source: obsolete expectation that absent optional stage history is rejected by writer. Exact-head reproduction is retained. Updated fixture tests actual acceptance/write/read, empty stage preservation, unchanged solid conservative energy and legacy NaN/Inf support roundtrip. Updated script PASS.
- Current material wrapper test independently exercises new argument chain without needing unavailable old objects; receipt follows.
- `git diff --check`: PASS at export.

## Appropriate budgets, not a fabricated momentum success

The pressure-history interface proposal is canceled and not implemented. The former 1.578e-4 whole-domain wet P residual is not repaired, reclassified as a pass or hidden by changing a tolerance. The reduced material model does not evolve solid/film momentum and no longer computes its unused support-force proxy. Valid physical validation remains evolved gas momentum versus actual gas face fluxes, mass/species/element and energy balances including actual phase and reduced-film work, and actual geometry/front consistency. All-participant physical momentum/support-reaction closure is NOT established. Existing independent physical-P failures/restart gaps remain qualified rather than converted to pass.

Final receipts: current material wrapper test PASS (1.13 s). After another worker repaired the test-only Thrust Policy allocator-call shim, mechanical-pressure backend singleton rerun PASS (8.54 s); original failed receipt remains for provenance. This leaves the unavailable old 360bf94 historical comparison BLOCKED and native OpenFOAM tests SKIPPED. No native CUDA execution was performed by this worker.

## Expanded legacy-audit cleanup, 16:50 UTC

A repository-wide source search found the five `Budget.turbulence*` statistics have no remaining physical consumer after the SST collector deletion; only schema, tests and validBudget referenced them. Removed those statistics from validBudget, preserving physical budget checks. Removed the explicit per-material-window assignments from geometryEndpoint.gasChemistryAudit/gasSstAudit: the material candidate already starts from base, so legacy payload remains unchanged without a second old vector copy. New focused test PASS (0.51 s): NaN/Inf legacy SST values neither block nor change actual packet mass/species/energy/momentum/counters versus a clean budget. Native material fixture now seeds nonempty/NaN base legacy and distinct endpoint legacy, then asserts base preservation alongside existing physical results; NOT RUN here because OpenFOAM is unavailable. This does not eliminate unavoidable whole-HostState legacy copies; removing the binary fields would require a separate format change.
