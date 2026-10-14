# Independent FAST3 delta review

PR14_REVIEW_ONLY · FAST · 2026-10-14 UTC. This is incremental source, test-support and claim review, not a new GPU result or whole-PR approval.

## Frozen scope and conclusion

- Baseline: `f5e6c5e2cbde2693f5fa811d874c200b07bbbc66`.
- Local source commit reviewed: `0643a7a40767b91896a87af9d06efb042082debd`.
- Published source/test commit: `9d5be6eff69dbced2fc959d422d3ba419a1bc39e`; publisher verified tree `5dc844d4189f50fcfbed2a77fa5e6cd6e9cfd066` exactly equals the local reviewed tree.
- Scope: one production file and four test files, enumerated with blobs and SHA256 in `../SOURCE.json`.
- No blocking source finding identified in this delta. `git diff --check` passed. No README, production chemistry, compensated-clock formula, kernel, traversal or physical-precision change is included.

## GPU-F006 production change

`applications/CHMT/materials/CpuMaterialDriver.C:461–463` evaluates zero phase mass as a zero recession-volume contribution without reading the unselected condensed-material index. The conditional is lazy. Material recovery, heat/force packets, film forcing, accounting and the original nonzero density/porosity expression remain in place. It does not introduce a rejection, rollback or stop boundary.

The new probe links and executes the real native CPU material driver and its film/surface/geometry dependencies with Foundation OpenFOAM and `-fsanitize=bounds -fno-sanitize-recover=bounds`. It is not a mocked driver or CUDA substitute. All three final modes passed: unselected zero mass, selected zero mass and valid nonzero melting. The independent reviewer reran each built probe mode successfully using the sourced OpenFOAM environment.

The archived original driver hash was independently checked against `git show f5e6c5e2:applications/CHMT/materials/CpuMaterialDriver.C`: both are `170ae1974fe71d14b093d3f39d93276cfb083cfe430149bbb9c1074a2757b79c`. Against the identical final probe, original unselected mode reports the index −1 bounds error, while selected and melting succeed. Their hexadecimal mass/heat/force/sweep output lines exactly match the repaired driver. The retained build/run logs under `native_material/` carry this evidence. The earlier incomplete fixture attempts remain separately labeled.

This is scoped native host verification. It does not close the formal GPU wet heat/evap, wet restart, nonzero phase GPU or complete-inventory requests.

## Chemistry fixture review

`tests/fixtures/chemistry_trial/chemistry_trial.cu.in` now keeps the canonical owner in ordinary host memory and allocates a separate device view. The device view is published after initialization and again after failure-matrix configuration changes. Kernel calls use the device pointer. Managed-data comparisons occur after synchronization; host launch metadata no longer aliases managed device storage. Normal cleanup releases both trial snapshots, the host ledger/owner, and tracked CUDA allocations.

The owner uses the actual native Courant preparation at unchanged target 0.5. Successful committed durations are captured from the real requested-interval owner and replayed on a separately initialized reference using native `true`. This tests accepted-state/rollback equivalence and exact requested-duration coverage. It is not a self-comparison and does not force two accepted parts. Separate public `false` controls cover terminal completion and the original full-owner success; the original direct chemistry success control also remains.

All 16 original failure-configuration prefixes and controls are preserved. Comparison code covers every semantic `ChemistryAudit` and `ChemistryStatus` member, cell/face transaction state, status arrays and budget channels. Host budget state and serialized records are compared, excluding only measured host wall-clock seconds. Structure padding is not treated as semantic data.

Independent command: `/tmp/pr14_validation_venv/bin/python -m pytest -q tests/test_chemistry_trial_cuda_fixture.py -k 'not compiles and not real_gpu'`. Result: `3 passed, 2 deselected in 0.07s`. These are generation/support checks, not CUDA runtime validation.

Fresh whole-production native chemistry-fixture compile/link exited 0 under CUDA 13.0.48 with `-O3 -arch=sm_89 --fmad=true`. The receipt, unchanged-source checks and actual binary hash were independently inspected. Binary SHA256: `19c1a37776d73a07edfa6c9a056d8880cd7cde61ca3533a7d81502d173e49b91`. Existing compiler warnings were retained. No GPU execution, WSL rerun or compute-sanitizer run was performed.

The fixture's `check`/`cudaCheck` helpers terminate on failure. Normal cleanup review is not allocation/free/sync error-recovery coverage.

## Native application compilation claims

Both Ns2 and Ns10 receipts show exit-0 native compilation of eight host source files and final CHMT application links with unchanged prior `Backend.o` objects. For each configuration, all 456 captured host-source/dependency hashes matched current files. The two reused object hashes and two final executable hashes matched their receipts on independent inspection.

The correct claim is native host compilation plus final application link with a hash-qualified reused CUDA object. It is not a fresh complete CUDA application rebuild or application execution.

## Prose review and remaining work

A draft wording mismatch in `REVIEW_REQUEST.md` described native accepted-duration replay as public `false`; it was reported and corrected to native `true`, keeping the separate public controls explicit. The reviewed reuse/gap plan retains original FAIL/BLOCKED/NOT_RUN records, distinguishes prior-data analysis from new runs, and requests bounded selective rechecks rather than rerunning unaffected evidence.

Required formal proofs remain: original wet/evap zero-phase behavior on the formal GPU application, valid nonzero-phase GPU control, and the repaired submitted chemistry fixture on a real GPU. Independent coverage gaps, original strict momentum/restart results, active cold groups and the two prior 300-second timeouts remain as specified in the delivery. No merge-ready, CI or merge-authorization claim is made. Administrative registration and its validator result are separate records.
