# Affected-suite failure triage

Baseline: `cc392dd40e5736494975163cad5b3139dcf1065b`. Baseline source was extracted using `git archive` into a scratch directory, without altering the shared development tree.

Original broad run: **961 passed / 228 failed in 519.10 s**. The following analysis separates exact reruns from representative sampling; it does not claim that the full baseline suite was rerun.

## Cache fixture updates

**170 failures** came from the two old point-cache/seed-reuse fixture files. Six representative old point-cache cases were run against the unchanged baseline and all six failed. Causes: T=199 K is supported property extrapolation, finite Y=-0.1 is clamped, the old uncached work count 2826 no longer matches 3582, and T=4000 no longer forces a retry. Current fixtures use a truly invalid T=-1 and NaN species, preserve all bitwise cache/reference equivalence checks, and check the work reduction rather than one obsolete reference call count. The full updated pair passed **698 tests in 5.42 s**.

## Exact baseline comparison for remaining failures

All **58 remaining original failing nodes** were rerun individually against unchanged baseline sources: **55 failed / 3 setup errors in 34.58 s**. There were **zero nodes that passed baseline but failed the broad development run** in this selection. This does not prove every failure has identical internal cause; the logs retain both diagnostics for review.

| Test file | Failing nodes in original run | Baseline outcome |
|---|---:|---|
| `validation/tests/gas_wall/test_wall_execution_work.py` | 1 | All fail/error |
| `validation/tests/gas_wall/test_wall_feasible_derivatives.py` | 1 | All fail/error |
| `validation/tests/gas_wall/test_wall_finite_publication.py` | 6 | All fail/error |
| `validation/tests/gas_wall/test_wall_geometry_single_construction.py` | 1 | All fail/error |
| `validation/tests/gas_wall/test_wall_jacobian_scale.py` | 1 | All fail/error |
| `validation/tests/gas_wall/test_wall_local_efficiency.py` | 2 | All fail/error |
| `validation/tests/gas_wall/test_wall_newton_globalization.py` | 1 | All fail/error |
| `validation/tests/gas_wall/test_wall_original_search.py` | 3 | All fail/error |
| `validation/tests/gas_wall/test_wall_shared_evaluation.py` | 1 | All fail/error |
| `validation/tests/gas_wall/test_wall_terminal_completion.py` | 18 | All fail/error |
| `validation/tests/gas_transport/test_boundary_layer_continuation.py` | 3 | All fail/error |
| `validation/tests/gas_transport/test_boundary_layer_host.py` | 1 | All fail/error |
| `validation/tests/gas_transport/test_boundary_layer_mechanical_pressure.py` | 1 | All fail/error |
| `validation/tests/test_sst_courant_mass_flux.py` | 1 | All fail/error |
| `validation/tests/test_sst_flux_consistency.py` | 4 | All fail/error |
| `validation/tests/test_sst_high_re_omega.py` | 1 | All fail/error |
| `validation/tests/test_sst_internal_diffusion.py` | 1 | All fail/error |
| `validation/tests/test_sst_low_re_omega.py` | 1 | All fail/error |
| `validation/tests/test_sst_owner_lifecycle.py` | 10 | All fail/error |

## Interpretation

- `test_boundary_layer_host.py` references the removed `minimumSubstep` member and assumes requested-interval retries. The baseline implementation deliberately uses an empty compatibility token and one requested trial; changing just the member access would retain an obsolete behavior assertion. The new `test_boundary_layer_stage_gradients.py` instead executes the current production stage body and verifies fresh RK gradients and first-profile reuse.
- Terminal/finite-continuation tests expect prior continuation or clipping behavior for invalid states and incomplete diagnostic mocks. These were not altered merely to make the suite green.
- SST compile/source assertions are already broken in baseline; examples include initializer mocks missing `gasProducerScalarValid` and old source substrings. The broad run is therefore not evidence of a new SST compile regression. New actual-adapter/full-tensor/blowing regressions are separate from these legacy fixtures.
- Fixed Newton iteration counts, exact historical captured heat output, old row reference implementations, and old instrumentation markers also fail on the baseline. Performance work should retain physical/convergence and equivalence checks appropriate to the final implementation.
- The diagnostics catalogue source-line assertion was already stale at baseline. The root agent is updating the catalogue after the production edits stabilize.

## Evidence

- `affected.log` and `affected.xml`: original broad development run.
- `baseline-triage.log`: six unchanged-baseline cache representative failures.
- `cache-fixture-triage.log`: updated cache/seed fixture pair, 698 passed.
- `baseline-other-failing-ids.txt`: exact 58 selected node IDs.
- `baseline-other-failures.log` and `.xml`: unchanged-baseline rerun of all 58.

## Exact selected-node outcomes

| Node | Baseline |
|---|---|
| `validation.tests.gas_wall.test_wall_execution_work::test_accepted_candidate_reuses_residual_and_table_preflight` | failure/error |
| `validation.tests.gas_wall.test_wall_feasible_derivatives::test_newton_trial_cannot_hide_selected_missing_thermo_segment` | failure/error |
| `validation.tests.gas_wall.test_wall_finite_publication::test_finite_continuation[signed_point]` | failure/error |
| `validation.tests.gas_wall.test_wall_finite_publication::test_finite_continuation[finite_input]` | failure/error |
| `validation.tests.gas_wall.test_wall_finite_publication::test_finite_continuation[zero_shear]` | failure/error |
| `validation.tests.gas_wall.test_wall_finite_publication::test_finite_continuation[signed_diffusion]` | failure/error |
| `validation.tests.gas_wall.test_wall_finite_publication::test_unrepresentable_newton_step_keeps_evaluated_profile` | failure/error |
| `validation.tests.gas_wall.test_wall_finite_publication::test_double_zero_shear_preserves_unavailable_diagnostic_and_real_fields` | failure/error |
| `validation.tests.gas_wall.test_wall_geometry_single_construction::test_finite_projection_budget_reports_residual_without_restarting` | failure/error |
| `validation.tests.gas_wall.test_wall_jacobian_scale::test_sqrt_k_derivative_has_a_resolvable_physical_scale[94]` | failure/error |
| `validation.tests.gas_wall.test_wall_local_efficiency::test_layer_row_skips_already_checked_unused_endpoints[14]` | failure/error |
| `validation.tests.gas_wall.test_wall_local_efficiency::test_layer_row_skips_already_checked_unused_endpoints[17]` | failure/error |
| `validation.tests.gas_wall.test_wall_newton_globalization::test_captured_sst_input_converges_in_physical_domain` | failure/error |
| `validation.tests.gas_wall.test_wall_original_search::test_original_successful_rejected_trials_and_complete_profile[reacting-one-damping]` | failure/error |
| `validation.tests.gas_wall.test_wall_original_search::test_original_successful_rejected_trials_and_complete_profile[reacting-six-damping]` | failure/error |
| `validation.tests.gas_wall.test_wall_original_search::test_original_successful_rejected_trials_and_complete_profile[sst-fifty-six-damping]` | failure/error |
| `validation.tests.gas_wall.test_wall_shared_evaluation::test_seed_reuses_only_invariant_thermodynamics` | failure/error |
| `validation.tests.gas_wall.test_wall_terminal_completion::test_terminal_wall_completion[analytic_trace]` | failure/error |
| `validation.tests.gas_wall.test_wall_terminal_completion::test_terminal_wall_completion[analytic_heat]` | failure/error |
| `validation.tests.gas_wall.test_wall_terminal_completion::test_terminal_wall_completion[point_state]` | failure/error |
| `validation.tests.gas_wall.test_wall_terminal_completion::test_terminal_wall_completion[point_rate]` | failure/error |
| `validation.tests.gas_wall.test_wall_terminal_completion::test_terminal_wall_completion[layer_diffusion]` | failure/error |
| `validation.tests.gas_wall.test_wall_terminal_completion::test_terminal_wall_completion[bvp_rate_tail]` | failure/error |
| `validation.tests.gas_wall.test_wall_terminal_completion::test_terminal_wall_completion[bvp_trace]` | failure/error |
| `validation.tests.gas_wall.test_wall_terminal_completion::test_terminal_wall_completion[bvp_diffusion]` | failure/error |
| `validation.tests.gas_wall.test_wall_terminal_completion::test_terminal_wall_completion[partial_enthalpy]` | failure/error |
| `validation.tests.gas_wall.test_wall_terminal_completion::test_terminal_wall_completion[precision]` | failure/error |
| `validation.tests.gas_wall.test_wall_terminal_completion::test_exchange_completes_all_actual_channels[heat]` | failure/error |
| `validation.tests.gas_wall.test_wall_terminal_completion::test_exchange_completes_all_actual_channels[velocity]` | failure/error |
| `validation.tests.gas_wall.test_wall_terminal_completion::test_exchange_completes_all_actual_channels[traction]` | failure/error |
| `validation.tests.gas_wall.test_wall_terminal_completion::test_exchange_completes_all_actual_channels[species]` | failure/error |
| `validation.tests.gas_wall.test_wall_terminal_completion::test_exchange_completes_all_actual_channels[omega]` | failure/error |
| `validation.tests.gas_wall.test_wall_terminal_completion::test_exchange_completes_all_actual_channels[area]` | failure/error |
| `validation.tests.gas_wall.test_wall_terminal_completion::test_exchange_completes_all_actual_channels[partial_h]` | failure/error |
| `validation.tests.gas_wall.test_wall_terminal_completion::test_exchange_completes_all_actual_channels[missing_h]` | failure/error |
| `validation.tests.gas_transport.test_boundary_layer_continuation::test_actual_donors_publish_complete_stage[False]` | failure/error |
| `validation.tests.gas_transport.test_boundary_layer_continuation::test_actual_donors_publish_complete_stage[True]` | failure/error |
| `validation.tests.gas_transport.test_boundary_layer_continuation::test_original_species_scaling_overflow_preserves_complete_exchange` | failure/error |
| `validation.tests.gas_transport.test_boundary_layer_host::test_actual_trial_reuses_first_profile_and_retries_large_outer_interval` | failure/error |
| `validation.tests.gas_transport.test_boundary_layer_mechanical_pressure::test_nonfinite_matching_pressure_completes_actual_force_and_work` | failure/error |
| `validation.tests.test_sst_courant_mass_flux::test_mass_only_matches_full_operator` | failure/error |
| `validation.tests.test_sst_flux_consistency::test_actual_sst_operators` | failure/error |
| `validation.tests.test_sst_flux_consistency::test_all_sst_gradient_consumers_have_same_stage_mass_flux` | failure/error |
| `validation.tests.test_sst_flux_consistency::test_courant_scratch_no_longer_overwrites_sst_mass` | failure/error |
| `validation.tests.test_sst_flux_consistency::test_volume_source_validates_before_commit_and_skips_disabled_arrays` | failure/error |
| `validation.tests.test_sst_high_re_omega::test_high_re_omega_stage_constraint` | failure/error |
| `validation.tests.test_sst_internal_diffusion::test_sst_internal_diffusion_and_stability` | failure/error |
| `validation.tests.test_sst_low_re_omega::test_low_re_omega_cell_constraint` | failure/error |
| `validation.tests.test_sst_owner_lifecycle::test_late_sst_canonical_owners_and_actual_initializer[CHT]` | failure/error |
| `validation.tests.test_sst_owner_lifecycle::test_late_sst_canonical_owners_and_actual_initializer[FSH]` | failure/error |
| `validation.tests.test_sst_owner_lifecycle::test_old_complete_first_repeat_exact_fields_and_work[CHT]` | failure/error |
| `validation.tests.test_sst_owner_lifecycle::test_old_complete_first_repeat_exact_fields_and_work[FSH]` | failure/error |
| `validation.tests.test_sst_owner_lifecycle::test_every_allocation_copy_launch_sync_failure_retries_actual_tail[CHT]` | failure/error |
| `validation.tests.test_sst_owner_lifecycle::test_every_allocation_copy_launch_sync_failure_retries_actual_tail[FSH]` | failure/error |
| `validation.tests.test_sst_owner_lifecycle::test_bad_numbers_use_real_initializer_and_missing_input_stays_error[CHT]` | failure/error |
| `validation.tests.test_sst_owner_lifecycle::test_bad_numbers_use_real_initializer_and_missing_input_stays_error[FSH]` | failure/error |
| `validation.tests.test_sst_owner_lifecycle::test_zero_size_does_not_add_publication_and_keeps_launch_error[CHT]` | failure/error |
| `validation.tests.test_sst_owner_lifecycle::test_zero_size_does_not_add_publication_and_keeps_launch_error[FSH]` | failure/error |
