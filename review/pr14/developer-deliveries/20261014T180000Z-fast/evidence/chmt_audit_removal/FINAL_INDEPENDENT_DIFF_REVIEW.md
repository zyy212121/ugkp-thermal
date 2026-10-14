# Independent final common/gasUGKP diff review

Read-only review, 2026-10-14. No production/test file edits or new execution tests were performed in this review. The earlier CHMT compile receipts remain separate evidence.

## Finding: unversioned V1 diagnostic wire shrink

The first integrated diff changed `boundaryLayerDiagnosticScalars` from 25 to 16 and moved species/reaction values from offsets 25/25+Ns to 16/16+Ns while retaining protocol major 7, opcode 28 and the `DownloadBoundaryLayerV1` API name. The same-build client/server/CSV mapping was internally consistent, but mixed binaries were not:

- Old client + new server: old expected size exceeds new reply, so `receiveResponse` returns `short GPU backend response`.
- New client + old server: `receiveResponse(requireExact=true)` only checks for a *short* reply, not equality. It accepts the longer old reply, reads a prefix, and discards the remainder. Old audit columns become new species values; multiple-wall record boundaries are also wrong. This can silently corrupt reported data.
- Direct old C-API consumers retain their old offsets and similarly misread the changed layout.

This is an observed source-level compatibility defect, not an assertion about numerical field evolution. Parent accepted the finding and assigned a codec-only correction: retain V1 25+2Ns rows and original species offsets, with the obsolete nine slots zero/reserved. The CSV can omit those slots without changing the wire. No audit accumulation or source-array readback needs to return.

## No additional physics/solver regression found in reviewed changes

1. Common gas flux divergence still accumulates the actual signed mass/momentum/energy/species face fluxes into physical update variables. Only `gasBudgetFlux` writes were deleted. The deleted helper's definition wrote solely to accounting storage.
2. Chemistry still calls the real reactor, publishes candidate species densities, recovers primitives and records real status. `gasBudgetChemistry` wrote only external accounting storage; its removal does not remove the species publication loop. Both source halves still receive `dt*0.5` and the policy's pre/post stage-volume pointer.
3. SST wall omega projection, k/omega floor/recovery, physical source/flux updates, source stability bound, conservative save and RK blend assignments remain. Removed beforeK/beforeOmega temporaries fed only the deleted audit helpers. The deleted preparation kernel wrote only `gasSstAudit.volume`, not physical `V` or ALE volumes.
4. The earlier `GasDevice` alias regression is fixed: `prepareGasTrialTransport` defines it at function scope before its remaining species capability conditions. No remaining call/include of the deleted SST/GasBudget helper names was found in common, gasUGKP production headers/backend, or gpu production code searched.
5. Native rollback cell snapshot width remains exactly `nCells*(21+compiledGasSpecies)`: 5 conserved gas + 5 primitive + Ns species + 3 thermodynamic + 3 step-bound + 5 SST fields. The six face boundary fields remain. Actual status arrays, chemistry status, wall exchange/SST/input/output/status records and rollback failure handling remain. Only audit channels were removed from snapshots.
6. `wallAuditStart/wallAuditInterval` were correctly retained despite their names: they restore wall `preparedTime/preparedInterval`, which the real preparation-reuse predicate consumes. They are not removable purely by naming them an audit.
7. Shared model reconfiguration still stages/uploads physical species/thermo/mechanism state, transfers ownership, republishes the view, and disposes the old owner. Deleted `GasBudgetRuntime` callbacks supplied reporting metadata, audit allocations and ledger I/O; they did not supply physical fields.
8. Boundary-layer heat flux, traction, wallKFlux, integratedKSource, ownerOmega, residual/iteration information, trace density/viscosity/stage time and species/reaction output were retained. The physical wall status/readiness checks were not confused with removable audit ledgers.
9. `BoundaryLayerConfigV1` retains its original size and offsets; the old budget switch slot becomes reserved and is encoded as zero/ignored on decode. It cannot affect active physics.

## Limits

This is a static diff/data-flow review, not a full native gasUGKP build, GPU execution, fault-injection matrix, ABI compatibility test or conservation proof. No new test workloads were started. Existing CHMT host parity/native support-compile evidence does not establish native gasUGKP runtime behavior. The V1 codec correction requires a final source check, recorded below when available.

## Codec correction recheck (17:01 UTC)

The shared worker's correction is present and closes the identified source-level mismatch:

- Protocol width is again exactly `25+2*species`, with unchanged opcode and request size.
- Backend result vector is value-initialized; only physical columns 0..15 and species/reaction columns 25.. are assigned. Reserved 16..24 stay zero; no six-channel D2H or audit calculation was restored.
- New CSV writer skips 16..24 while retaining row stride `25+2*species`; its header has the same 16+2*species emitted columns. It reads original species/reaction offsets even from an old server whose retired columns are nonzero.
- Old client receives the original byte size and species offsets from the new server. Its existing `budgetAuditAvailable` column sees zero, causing old CSV fields 17..24 to be blank rather than report zero-valued measurements. New CSV has no audit labels at all.
- Multiple walls use the same unchanged wire stride on both sides, so the previous row-shift issue is also removed.
- Existing worker tests now contain an old-server-shaped payload mock (including numbered nonzero retired columns) expecting species offsets 25/28, plus new-server storage assertions that all nine reserved words are zero. This review inspected those assertions but did not run or claim their results.

Disposition: one source-level IPC compatibility finding, corrected and independently rechecked. No further concrete physical update, dt, rollback ownership, remaining-deleted-symbol or necessary-field omission found in this bounded final review. Mixed-version executable tests and full native gasUGKP runtime remain outside this review's evidence.
