# Five chemistry tests: removed-audit API migration

Exclusive edits: test_gas_finite_continuation.py, test_gas_publication_limits.py, test_gas_terminal_completion.py, test_rounded_chemical_publication.py, test_gas_budget_publication.py. Shared adapter SOURCE is owned separately; no production edits here.

## Baseline

Exact original HEAD source and tests at /tmp/pr14-sst-head-regression: 1 failed, 59 passed (28.66s). The failure is test_adapter_publishes_quality_diagnostics: stale.cellStatus==0 && stale.chemistryStatus assertion. HEAD.log preserves the complete evidence. This expectation is not silently changed during API migration.

On current production before migration: 9 failed, 51 errors, all blocked by deleted audit API/fields (before.log). First migrated iteration: core 43 passed, remaining 17 blocked by shared adapter SOURCE awaiting its separate migration (after.log).

## Replacement coverage

- finite_continuation: actual species mass residual and independent element inventory replace output audit residuals. Attempted-step control and finite temperature remain; adapter density versus direct reactor and unchanged energy remain.
- publication_limits: individual finite published fields and exact unmodified conversion remain. The former output-audit-overflow case calculates the overflowing sum/product within the test only; it is overflow propagation coverage, not proof of physically conservative input. Large finite publication still matches the direct reactor.
- terminal_completion: original actual field and nonfinite propagation checks remain. Full/half/end manually executed stages still match actual reactor mass/temperature; attempted/rejected controls remain. Two real adapter half-steps use their respective volumes, compare to separate direct reactor predictions, and independently check signed mass changes, mass conservation, and unchanged energy.
- rounded_chemical_publication: exact direct-reactor conversion/repair comparisons and unchanged bulk/other-cell fields remain. Actual stored masses and element changes are recomputed in long double, with 1e-12 mass and 1e-10 element residual bounds; stored masses times species energy at the direct reactor temperature must match input energy to 1e-7. These tests intentionally include tighter user-control cases where the direct continuation path still publishes; test bounds do not add production gates.
- budget_publication: before/after stored density times actual volume, unchanged energy, exact zero-step storage, and missing-volume behavior remain. Zero-time actual inventories are measured in the test.

Removed acceptedSteps, integratedTime, maximumLocalError, and nonlinearIterations output claims are no longer tested as production observations. Interval correctness is covered by actual state comparisons at specified stage intervals, not by fabricating replacement metadata. NaN/Inf scenarios are explicitly propagation coverage, not valid physical-solution PASS claims.

Host execution only. No additional native CUDA compilation or GPU execution is claimed by this test migration.

## Final execution and retained failure

Final five-file suite: **1 failed, 59 passed in 31.34s**, exit 1. Exact same failed test and assertion as HEAD. final.log/final.exit preserve results. The first full iteration (first-complete.log) additionally caught my new two-half test incorrectly treating the initial second species as zero; the fixture is 0.5. This was fixed by capturing both actual initial densities before computing differences, without relaxing tolerances or changing production.

The retained stale-status assertion is not cross-case fixture pollution: it creates a fresh `Fixture stale`, explicitly writes NegativeInventory to aggregate cellStatus and ConservationFailure to chemistryStatus, then calls the low-level direct adapter. The successful adapter overwrites chemistryStatus but does not reset the aggregate cell status (advanceGasChemistryKernel.cuh:245); original HEAD behaves the same. Real owners reset statuses at trial entry: CHMT Backend.cu:117 -> SharedGasDeviceStorage.cuh clearTrialStatus; gasUGKP SharedGasTrialPolicy.cuh resetErrors clears cell, face and chemistry statuses. Its transportStatus reader considers nonzero cell/face status only in native-refinement mode or for InvalidStorage; CHMT checkStatus rejects InvalidStorage. The test's expectation that the adapter independently clears a deliberately injected aggregate status is a separate old contract, not evidence of a new physical regression. It remains failed rather than silently resetting input or reversing assertions. Assertions later in that same C++ executable are not reached by this failed run.

Final patch and selected source/dependency SHA256s are tests.patch and source-sha256.txt. git diff --check passed for the five files.

## Authorized owner-contract migration

After the parent confirmed the real owner-reset semantics, the old stale test was split into two parameterized cases. `direct_preserves_aggregate` injects the prior NegativeInventory and requires it to remain while chemistryStatus reports the real successful solve. `owner_reset_success` performs the same status reset as trial entry before the call and requires success with cleared aggregate status. Both compare all actual species densities to a fresh successful reference solve, check species conversion direction and total density, and check unchanged energy, density and momentum. Neither introduces production behavior. The original HEAD-red and intermediate retained-failure logs remain.

Final owner-contract suite: **61 passed in 33.35s** (owner-final.log/owner-final.exit). No original failed assertion was simply removed without replacement coverage. Later assertions in the C++ adapter executable now execute successfully in both scenarios.
