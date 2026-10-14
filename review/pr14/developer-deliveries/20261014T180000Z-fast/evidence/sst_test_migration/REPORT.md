# Shared SST test migration after runtime audit deletion

## Ownership and changes

This worker exclusively updated ten files listed in source-sha256.txt. Initially assigned two additional files (test_bulk_review_findings.py and test_boundary_layer_state.py) were handed to audit_redundant_runtime_work after exact-HEAD evidence showed 19 existing assertion failures; they are deliberately excluded from tests.patch. No production source changes were made in this test-migration task.

The shared `AUDIT` import name is kept for fixture callers, but its contents are now test-owned `SstObservation` storage, not a State member or a production interface. Before each real Euler SST update, the observer independently traverses the supplied incidence, calculates signed actual face impulses, absolute ALE volume divergence and source terms, predicts the physical k/omega inventory including explicit constraints/floors, then calls the unmodified production updater and compares its actual output. Nonfinite numerical-propagation cases check matching IEEE output AND existing status in their specific test; they are not evidence of a valid physical solution. Finite expected values reject nonfinite actual output. Strict calls bypass this observer and invoke the original strict updater immediately, so no extra test-side backing access or pre-status evaluation can alter strict tests.

RK tests retain independently weighted test-side transfers and compare actual integrated cell inventories; physical save/blend, bulk momentum, species and energy assertions remain. Recovery additionally checks that k changes only by its physical floor; actual wall omega target assertions remain in the wall tests. A test-only mutation of the actual production kernel source doubles its k source contribution; the test requires the independent physical predictor to reject it. Thus this migration does more than delete missing-field assertions.

Removed old gas-budget setup/assertions from cell/bulk fixtures, keeping actual conservative-state and strict retry tests. A former six-channel ledger-sentinel test now checks actual k/omega/momentum SSPRK weights during species-NaN continuation. Current-Co fixture now imports only its local function extractor and uses actual production reduction structs with updated Thrust/test adapter stubs, preserving four-copy restore and error-path checks. The mass-only Courant test's helper signatures match the current production calls; the missing-owner negative case uses the real strict path, rather than executing a terminal unbacked negative index. It retains all nine Riemann schemes, reconstruction/boundary combinations, mass-only/full parity and laminar/SST scratch checks.

## Original failures and migration evidence

- pr14-sst-migration-red.log: old shared helper cannot compile after production audit removal, expected red (`State` has no gasSstAudit).
- pr14-sst-migration-suite.log: first migrated 12-file run: 162 passed, 29 failed; complete output retained, not replaced by the later result.
- pr14-sst-old-obsolete-suite.log: exact HEAD common+tests (0aa34bc source tree) reproduces the same 19 failures in the two handed-off files: 19 failed, 26 passed.
- pr14-sst-old-co-face.log: exact HEAD reproduces old expectation of OfGreat for terminal NaN geometry. Production already retains evaluated NaN spectral/mass values and records failure; updated test checks those diagnostics, not physical validity.
- pr14-sst-old-co-wave.log: exact HEAD reproduces old two-NaN wave fallback expectation. Current production intentionally uses OfGreat only in strict mode; terminal mode carries NaN and failure status. Migrated test explicitly covers BOTH modes while preserving the finite fmax case.

No numerical tolerance, solver admission, actual source/flux arithmetic or strict failure behavior was changed.

## Results

- Earlier ten-file full run: 148 passed in 95.99 s (before adding the extra mutation test).
- Production-source mutation sensitivity: 1 passed in 0.71 s (the intentionally wrong source was correctly rejected).
- Updated predictor plus wall/source/consumer group: 25 passed in 18.77 s, including mutation sensitivity.
- Bulk/cell terminal and strict group: 117 passed in 73.67 s.
- Mass-only Courant/full physical route: 1 passed in 1.32 s.
- A final frozen ten-file run (including mutation and immediate strict bypass) is running; its receipt will be attached separately.

These are real production C++ kernels executed through existing serial host CUDA shims. They do not establish native CUDA execution, GPU performance, global solid/film momentum closure, or native OpenFOAM integration.

Final frozen ten-file validation: **149 passed in 95.55 s**, including the independent source-mutation sensitivity test and direct strict-path preservation. Receipt: pr14-sst-ten-frozen-final.log. Owned-file `git diff --check` passes. No commit or push performed by this worker.
