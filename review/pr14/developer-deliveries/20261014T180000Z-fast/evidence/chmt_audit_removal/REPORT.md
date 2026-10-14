# CHMT redundant runtime review and removal

Scope: PR14 checkout `pr14_fast_fix_20261014`, starting HEAD `0aa34bc`; work is uncommitted. This worker does not touch README, PRConversation, push or merge. Existing and concurrent support/film changes belong to other workers. The initially considered opt-in diagnostic build was superseded by the user's explicit instruction to remove pure audit work. No new compile flag remains.

## Confirmed and removed

### SST integral accounting: no physics/control consumer

Producer: common `GasSstAudit` helpers read actual `rhoK/rhoOmega`, transport/source increments, old/new volumes and wall projections. They write only six diagnostic channels, six RK initial copies and an audit volume. `prepareGasSstAuditKernel` sets that audit-only volume; `saveGasSstAudit`/`blendGasSstAudit` operate only on accounting arrays.

Former CHMT consumer: `acceptSstAudit` performed six synchronous full-cell D2H reads, six sums and an inventory sum. It accumulated `HostState.gasSstAudit` and `Budget.turbulenceInventory/turbulenceBoundaryFlux/turbulenceOmegaConstraint`. Material copied these values; checkpoint serialized them. Neither `IntervalAudit` nor coupling/time-step selection reads those six sums or the three budget fields. Legacy generic budget validation is being independently cleaned by the support worker.

Removal: CHMT owns no 13-array SST audit allocation, clear, swap, release, download or accumulation. Initially CHMT specialized the shared capability to false. The parent then authorized complete shared removal: the separate common worker removed the `GasSstAudit` type, device member, prepare kernel and helper calls, so this worker removed that temporary specialization too. The final CHMT view has no SST audit descriptor at all. No flag, mirror status, new precheck or traversal was introduced.

Removed steady-state cost: 13*Ncells*sizeof(double) device bytes; 13 cell-buffer memset calls per trial; audit-only volume preparation kernels; added accounting global loads/stores in the real physical kernels; six Ncell double D2H calls per committed microstep plus host sums. Exact speedup is unmeasured.

Preserved: actual SST state, initial RK state, conservative update, wall owner omega, k/omega constraints, eddy viscosity, source stability numbers, Courant preparation/reduction/selected dt, and all existing FP64 arithmetic order in common operators.

### Published chemistry audit: no physics/control consumer

Producer: shared `advanceGasChemistryKernel` previously saw a non-null external `chemistryAudit` pointer, recomputed rounded published species mass changes, total mass residual, elements and mixture energy for reporting, then stored one audit record per cell. Two trial callbacks downloaded pre/post chemistry records. CHMT commit summed them into its cumulative host diagnostic vector.

The shared kernel's local reactor `localAudit` is distinct: the internal reactor/retry method can use its conservation/error information to decide its solver result. It is preserved. Actual chemistry result, status recording, species densities, temperature recovery and source split volumes are unchanged.

Removal: CHMT external buffer allocation/lifetime/zero and both D2H reads are gone; `firstChemistry/secondChemistry` host vectors and commit accumulation are gone. The shared required `captureChemistryAudit` hook is a constant no-op. `gasSpecies.chemistryAudit` is null, so the reporting-only rounded-mass/element/mixtureEnergy branch is skipped. No shared solver arithmetic was edited.

Removed cost: Ncells chemistry audit GPU storage and clear; two full audit-record D2H copies per microstep; two host Ns reductions; published-inventory diagnostic recomputation in the kernel. No timing claim.

## Compatibility and outputs

`Checkpoint.C` field schema/order and bytes are unchanged by this worker. `HostState.gasChemistryAudit` and `gasSstAudit` remain legacy payload slots, not newly measured values. Their comments state that limitation. Output summary adds `sst_chemistry_integral_audit_collection=NOT_COLLECTED_LEGACY_CHECKPOINT_SLOTS_ONLY`, so zero or imported legacy payload is not presented as fresh measured accounting. Physical gas/material/exchange/wall-layer output writers remain present.

Support worker owns deletion of the explicit redundant material-window legacy copy and legacy validation. Remaining whole HostState snapshots are shared rollback/compatibility storage, not newly introduced diagnostic traversals.

## Other reviewed work: retain or optimize separately

1. `currentGasCourant` is NOT dispensable reporting: its reductions select `selectedDt`; SST state is saved/restored because Co preparation can project it. Removing it changes time partition and potentially every field. Keep.
2. `stageBulk/stageGradient/gasTrace/gasTraction` are NOT unused snapshots: `advanceGasMicrostep` selects interface owners, then `CpuMaterialDriver` and `MaterialDrive` consume traces/gradients/traction for the physical material solve and drive splitting. Keep values. `readBulk` currently downloads all cells and then keeps interface owners: sparse gather could remove excess transfer but requires actual interface/periodic/moving ownership tests and is not implemented here.
3. `accountCoupledFaces` final limited flux is NOT solely a report. Accepted packets drive CPU solid/film forcing; radiation has a physical receiver; boundary M/E/elements enter the coupling assessment and can change replay/shortening. Do not drop the full function. All-face D2H and interface boolean rebuilding are possible locality optimizations, not proof that fluxes are unused.
4. `auditCoupledCandidate` returns `assessment.conservative`; the controller can replay/shorten based on it. Thus it currently affects the executed trajectory even though it does not directly write a field. Any removal is a deliberate behavioral change, not observationally neutral dead-code deletion. Root has a separate audit classification worker.
5. `finalizeIntervalLedger` mainly assembles reporting/history records, but `exchange-*.csv` is a useful existing output and checkpoint preserves the ledger. Do not silently remove output functionality as an efficiency edit.
6. Stage geometry deep copies into `gasStages/solidStages` are only queried for optional geometry output and codec, whereas live `gasStage` old/new volumes and sweeps drive ALE kernels. The stored duplicate stage arrays may be materialized later, but removing physical live geometry is incorrect. Not changed here.
7. `sampleGasWallProgramLocal` is called by prepare/apply/account for the same stage. Reuse within a stage may remove duplicate interpolation/copies, but cache invalidation across preflight, selected dt, rollback and RK stages must be tested. Not changed here; no new cache.
8. `finishCoupledGeometry` uploads full geometry even for static microsteps. It also republishes boundary fields that GPU kernels may mutate. Skipping the whole routine merely because points are static is not yet proven equivalent. Not changed.
9. Device status synchronizations scan failures that trigger rollback. They are not treated as harmless logging and have not been removed without a consumption proof.
10. The existing boundary-layer stderr workspace line is configuration-time, not per-stage, and reports useful allocation choice. It was not targeted ahead of recurring full-domain copies.

## Other applications

`gasUGKP/private_backend/BoundaryLayerAdapter.cuh` allocates the same 13 SST arrays when `budgetAudit` is enabled and downloads six channels only into wall diagnostic columns 16..24. `SharedGasTrialPolicy.cuh` snapshots seven audit arrays for rollback. This is a related pure accounting path, distinct from real wall `ownerOmega/wallKFlux/traction`. Findings were delivered to root and the separate gasUGKP worker, who is implementing that cleanup and shared hook removal. This worker made no shared/application edits outside CHMT.

## Verification, precise scope

- `physical-parity.log`: PASS. A scratch probe builds HEAD Backend+Storage and changed Backend+Storage against the same other sources and host CUDA shim. Euler/RK2/RK3 each execute three microsteps. Exact hexadecimal gas M/E/P/species, rhoK/rhoOmega, accepted dt/next dt/time and exchange packet masses/energies/momenta/species match. This is host emulation, not a real GPU execution or whole-case conservation proof.
- `summary_metadata.cpp`: compiled/run PASS; metadata explicitly marks noncollection, supplied legacy payload remains unchanged.
- `focused-tests.log`: 7 passed; three failures retained. Two checkpoint historical compatibility tests cannot load pinned Git objects 3757ee.../40e1b4... in this checkout. The output writer test's old expectation that invalid material output must reject fails identically with original HEAD writer and original test (`output-baseline/result.log`, exit 1); it is not introduced by metadata.
- `storage-backend-final.log`: final focused rerun after correcting the legacy sentinel test (result appended later).
- `nvcc-Ns2.log` / `.exit`: real nvcc 13, sm_80, -O2 full Backend compile was killed, exit 137, no object. This is a failed production-option compile attempt, not a pass.
- `nvcc-Ns2-lowopt.log` / `.exit`: lower device optimization support compile running when this report was drafted. It must not be substituted for -O2 or GPU execution.
- Four historical audit consumer test files are being updated independently by the repair worker to observe real fields in test-owned hooks. No claim of their completion here.
- No new permanent source/test files were added by this worker, so no new ownership-manifest paths are needed. Existing production files and test files are modified only.

No GPU speedup, full original-case restart parity, or complete gas/film physical momentum closure is claimed. Deleting a nonphysical ledger does not establish physical conservation.

## Integration updates

- Initial CHMT-only deletion passed six backend/storage tests (`storage-backend-final.log`). Its low-device-optimization Ns2 native compile passed, with object SHA256 `0035be2c0d77a1e697570fc2aa2f3088b23dc6097a4ca3ebe22570c26876bbe8`; this was before final common hook removal and is explicitly intermediate evidence.
- First common integration exposed an accidentally removed `GasDevice` alias still used by real species capability checks in `prepareGasTrialTransport`. Three backend tests and Ns10 native compilation caught it; the common worker restored the alias. Failed logs are retained (`final-common-host-tests.log`, `nvcc-final-Ns10-lowopt.log`). Ns2 native compilation in that batch was killed (137), distinct from the syntax error.
- Final fixed-code host, physical comparison and sequential native compile checks are being completed; do not infer a final pass from intermediate evidence.

## Final receipt (2026-10-14 16:57 UTC)

- Final fixed shared/CHMT source: six backend/storage tests passed (`final-common-host-tests-fixed.log`, 26.56 s). Legacy imported nonzero audit payload was supplied before backend creation and remained unchanged; no SST device descriptor/array exists.
- Final physical comparison against the recorded HEAD baseline passed (`final-physical-parity-fixed.log`) for all previously listed hexadecimal fields, dt and packets.
- Real nvcc full Backend compilation with Ns2 and Ns10 both completed with exit 0 at low device optimization (`nvcc-final-fixed-Ns2-lowopt.*`, `nvcc-final-fixed-Ns10-lowopt.*`). Exact commands, source digest map, object digests and limits are in `final-nvcc-commands.json`, `final-production-source-sha256.json`, `FINAL_RECEIPT.json`. No GPU execution occurred. Production -O2 compile was not established; the original exit 137 remains a failure, not replaced by these support compiles.
- Code review by the support-removal worker found no new P0/P1 in this resource deletion and independently verified that no real source, step selector, profile storage or physics transaction was dropped.
- No source edits remain pending from this worker. Related test migration and full repository integration remain owned by the parent/other workers.
