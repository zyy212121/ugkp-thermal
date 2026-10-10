# Shared boundary-layer integration: evidence and cost review

## Scope and limits

The new family is explicit: `lowRe=0`, `wallFunction=1`, `boundaryLayer=2`.
`constantTransport` is a frozen laminar analytic limit. `finiteRate` (alias
`reactingSst`) uses shared species thermo/mechanism and optional shared SST.
The auxiliary solve owns no gas or chemical inventory. Bulk chemistry remains
the only reaction inventory update. This integration report does not certify
all wall-model mathematical cases or GPU runtime performance; those require
the core validation results and an actual GPU run.

Physical total species flux, conductive heat and traction are published as one
complete conservative wall exchange. Formation enthalpy is included once via
shared species enthalpy. Matching-plane species flux and reaction integrals are
diagnostics only. The entire selected owner's k source is replaced once by the
actual-volume integral. Omega transport/source channels retain the replaced
coarse discrete row; a finite owner constraint supplies the sixth-channel
closure, without integrating a singular physical wall omega flux.

## Stage lifecycle and necessary work

1. Geometry: CPU mesh extraction and descriptor building occur at configuration
   or geometry-version rebuild, not each stage. Maps, positive quadrature and
   matching stencils upload once. Selected wall owners cannot be matching donors.
   SST owners use profile-node/stretch-aligned positive section quadrature by
   default; laminar walls retain ordinary geometry quadrature. Its CPU work is
   initialization/rebuild work, and its device distance/weight storage costs
   `2*sizeof(Real)*sum(ownerQuadratureCounts)` in addition to face records.
2. Outer step proposal: prescribed Js/ALE mass can be evaluated without a BVP.
   A previously accepted profile source contributes to the preliminary SST bound;
   only the cold-start missing owner source is omitted from this estimate.
3. Trial: snapshot, first chemistry half-step, recovery, current material boundary
   input, one sparse profile kernel, then authoritative existing stability checks.
   Boundary mirrors and owner projection are moved into this preparation and
   reused by the first Euler stage, rather than repeated.
4. Euler/RK: first Euler consumes the exact generation/time/interval cache.
   Each later RK state gets a fresh profile. A changed material input explicitly
   invalidates it. New trials and rollbacks also invalidate it.
5. Physical positivity/source-step failures use the existing two-level retry.
   A pure BVP failure uses terminal `BoundaryLayerFailure`, not bulk chemistry
   failure and not repeated dt halving. Failure-only diagnostics give face,
   core code, node, iteration and actual residual (NaN if unavailable).
6. Accepted outputs and transaction snapshots include compact wall exchanges,
   inputs, outputs, statuses and stage time. Native gasUGKP SST budget diagnostics
   are explicit opt-in (`boundaryLayer { budgetAudit true; }`, default false).
   Only enabled audits snapshot six channels, six RK initial copies and volume.
   CHMT retains its pre-existing mandatory SST ledger; this feature does not add
   a new full-cell audit requirement to that adapter.

For a successful cold-start or subsequent trial, sparse profile launches are
Euler: 1, RK2: 2, RK3: 3. Cold start adds no speculative outer BVP. A rejected
trial necessarily prepares once again from the restored/recomputed state.
Executable scheduling tests verify these counts, chemistry-before-profile,
first-stage reuse, fresh RK/retry preparation and bounded large-step recovery.
The legacy boundary mirror count is 2 per trial including finalization; Riemann
mirror count is RK-stage-count + 1. Default native runs launch no audit-volume
preparation. Opt-in runs reuse it once per trial rather than twice. Turning the
native diagnostic audit off saves 13 double cell arrays plus their two existing
snapshot copies: 104 bytes/cell live + 208 bytes/cell snapshots = 312 bytes/cell.
It also removes one full-cell preparation launch and the corresponding snapshot
copies. Executable on/off tests retain identical physical k/omega results.

No new successful-stage host synchronization or field download is needed for
the native gas profile kernel: it uses the existing trial validation points.
Allocation/configuration queries and geometry downloads occur once. Diagnostic
CSV downloads happen at writes. Error detail downloads happen only on failure.
Transaction snapshots add device-to-device copies at the existing two snapshot
levels. Existing face/cell operators consume compact maps in their existing
traversals. Unselected lowRe/wallFunction runs allocate no wall storage and
launch no profile kernel. CHMT transfer details are recorded in its adapter
report; its material interface requires separate exchanges.

## Bounded scratch and parallelism

`constantTransport` has no Newton scratch allocation and uses one sparse
thread per wall face. Finite-rate Newton uses one thread per scratch slot,
with grid-stride reuse across wall faces and 32-thread blocks. This is not
block-cooperative: divergent iteration counts and dense serial solves can
limit GPU utilization; no measured GPU speedup or occupancy claim is made.

The allocation capacity is bucketed by configured nodes: <=24, <=48, <=128.
The workspace array has `capacity * (1 + 4*(Ns+4) + 8 + 3*(Ns+4)^2)` scalars.

| capacity | Ns2 FP32 | Ns2 FP64 | Ns10 FP32 | Ns10 FP64 |
| --- | ---: | ---: | ---: | ---: |
| 24 | 13536 | 27072 | 62688 | 125376 |
| 48 | 27072 | 54144 | 125376 | 250752 |
| 128 | 72192 | 144384 | 334336 | 668672 |

All sizes are actual `sizeof` bytes, recompiled at the conditioned-core snapshot;
the eight initializer scalars per node are included. `workspaceSlots=0` selects
`min(walls, 32*SMcount, 4096, min(freeMemory/8,1GiB)/workspaceBytes)`.
Positive slots (1..4096) are explicit bounded overrides, capped by wall count
and checked against reported free memory. Free memory is a snapshot, not a
reservation; allocation failure releases the candidate cleanly. Configuration
reports nodes, capacity, resolved slots, allocated bytes and budget. One block
per SM is only an initial auto policy, not a performance optimum. Workspace
memory is O(slots*capacity*Ns^2); per-face inputs/outputs remain O(walls*Ns).
The required face-slot and owner-slot index maps add
`sizeof(int)*(nFaces+nCells)` = `4*(nFaces+nCells)` bytes on this backend, separate
from per-wall records. They are initialized/uploaded once at configuration or
geometry rebuild and consumed in existing operators, without a new per-step
mesh traversal. Matching CSR and quadrature arrays add storage proportional to
the actual descriptor counts. No native resolved-slot count or GPU timing is
claimed in this CPU-only environment; configuration prints resolved slots and
bytes on the user's GPU for performance experiments.
Boundary-layer configuration must precede the first trial: enabling it after
existing trial/interval snapshots would change wall record storage and, when
requested, their SST audit field layout.
The public ABI now rejects that sequence without mutating existing storage.
SST must likewise be configured before enabling boundaryLayer. Subsequent
ConfigureSst calls are rejected while the new family is enabled, preventing a
selector change or k/omega reset from disagreeing with the retained profile
maps. Legacy residents still permit their previous repeated SST configuration.

## Verification and broad-suite comparison

Earlier integration-checkpoint shared/native-ownership command:
`../venv/bin/python -m pytest -q tests/gas_transport applications/gasUGKP/tests/test_boundary_layer_protocol.py applications/gasUGKP/tests/test_boundary_layer_storage.py applications/gasUGKP/tests/test_shared_gas_trial_policy.py`
completed with **172 passed**. The actual stage probes additionally poison all
workspace bytes before evaluation, matching raw CUDA allocation rather than
relying on constructor-zeroed scratch; all 15 stage/precision probes pass.

Native OpenFOAM configuration/geometry/recording-ABI/CSV tests: 23 passed,
including profile-aware SST owner quadrature, default auto0, explicit auto0 with 96 nodes, model restrictions,
physical-wall geometry and actual diagnostic column mapping, optional audit fields and blank unavailable
channels. This is a real
OpenFOAM frontend test with recording backend, not GPU execution.

Executable common tests cover FP32/FP64 physical exchange signs and formation
energy, actual analytic/finite-rate stage evaluation, owner source/constraint
six-channel budgets, fixed-Js rejection instead of partial limiter scaling,
source-step rejection before inventory writes, cold/accepted outer estimates,
storage failure cleanup, two-level wall/audit rollback and cache counts.

Production wall precision is fixed FP64 in both adapters: native gasUGKP uses
`GasBoundaryLayerModelState<double,...>` and CHMT's `Real` is `double`.
Generic FP32 core/operator probes are separate precision tests, not an available
native FP32 wall backend. There is no public wall-precision or automatic-tolerance
selection in this ABI. Explicit requested tolerances are never relaxed.
The conditioned core at a10fe74 uses relative temperature, tangential velocity
and a fixed energy reference internally; physical temperature and formation
energy retain their original definitions. Its FP32 accuracy/convergence evidence
belongs to the core validation report, independently of production FP64 binding.

### Optional owner-inventory budget CSV

The existing `owner` identifier refers to a unique selected owner in native
configuration (duplicate owners are rejected). These channels are not per-face
fluxes and must not be summed repeatedly for the same owner. `budgetAuditAvailable`
is 0 by default, with the eight following fields empty. When it is 1,
`budgetIntervalStart` and `budgetIntervalDuration` identify the last accepted
microstep, not an accumulated requested output interval. The six finite channels
are `budgetTransportK`, `budgetTransportOmega`, `budgetSourceK`,
`budgetSourceOmega`, `budgetConstraintK`, `budgetConstraintOmega`. K increments
are volume-integrated rho*k inventory changes in J; omega increments are
volume-integrated rho*omega changes in kg/s. Transport is positive inward,
sources are signed, and constraints include suppressed equations, floors and
projection. They follow the same SSPRK algebra as the inventories. Rollback
restores interval metadata with the corresponding outputs and channels. The
numeric IPC uses unavailable sentinels internally; the CSV writer emits blank
fields, never fabricated zero budgets or NaN text.

Broad command in both baseline 8313227 and wall worktrees:
`../venv/bin/python -m pytest -q tests/gas_transport applications/gasUGKP/tests`.
Baseline: 77 failures (57 test failures + 20 failed subtests), 383 passed,
29 skipped, 8 errors, 391 successful subtests.
Final conditioned-core/opt-in comparison run: the same 77 failures and 8 errors,
418 passed, 52 skipped,
391 successful subtests. The additional 23 skipped frontend tests were run
separately in the OpenFOAM environment and all passed. Exact failed identities,
including subtest tokens, have added=0 and removed=0. This is not a green broad
suite: existing source-layout/contract assertions and extraction errors remain
recorded below. They were not weakened or silently classified as passing.

The unchanged failures include source extraction of kernels now residing in
common headers, source spelling/templating assertions, and historical scheduling
and application-layout checks. Their pre-existence does not establish numerical
correctness of every affected subsystem. The focused production operator and
transaction tests are separate evidence. The two integration regressions found
in the initial broad run (capability message compatibility and optional wall
member classification) were fixed without relaxing the underlying legacy guard.

Native CUDA validation is compile-only: CUDA 13.0.88, sm_89, no attached GPU.
Early pre-capacity builds and interrupted intermediate builds are not final
validation. The isolated preconditioning resource experiment at source 4708cbc
used the unchanged release -O3 settings. Outlining only the capacity entry failed
with exit 137, 214.6 seconds and peak child RSS 4,823,888 KiB. Outlining the five
nested device helpers pointState, pointReaction, pointSst, intervalFlux and
layerRow as well succeeded in 227.1 seconds at 1,569,344 KiB. CPU inline semantics
and mathematical formulas are unchanged. Cgroup memory counters were unavailable,
so exit 137 alone is not treated as proof of an OOM kill.

For that successful isolated Ns2 build, ptxas reports the sparse wall kernel at
255 registers/thread, 3024 bytes cumulative stack, 20 bytes spill stores and
8 bytes spill loads. The direct PTX call graph has at most four edges and no
recursion. Helper stack frames are zero; the capacity wrappers still have spill
traffic (76/100 store/load bytes for 24/48; 92/276 for 128). These resources can
limit occupancy and make the one-thread-per-wall strategy costly. No performance
or stack-runtime safety claim follows from compilation; no manual large device
stack-limit override is introduced.

### Final native gas application receipts

Both species specializations completed from immutable committed source
`1f4cf01f83e3d2cbac16dc59dc4387f7a6a5dc60`, containing conditioned math a10fe74,
five outlined helpers e8ec607, capacity dispatch 7d11dff and opt-in audit 1f4cf01.
Each build compiled the actual full `GpuResidentStrict.cu` translation unit with
`-std=c++17 -O3 --fmad=true -arch=sm_89 -Xcompiler -fPIC`, then linked the private
backend server and compiled/linked the real OpenFOAM gasUGKP frontend with the
same species count. Both source-manifest rechecks passed. `ldd` found no missing
dependencies, no CUDA linkage in the frontend, and no OpenFOAM linkage in the
private backend. These are compile/link receipts, not device-execution results.

| Species | CUDA compile seconds | Peak child RSS KiB | Registers/thread | Cumulative stack B | Spill stores/loads B |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 227.9 | 1572032 | 255 | 3312 | 20 / 8 |
| 10 | 278.9 | 1929756 | 255 | 7680 | 44 / 16 |

The stack/register/spill columns describe the actual wall launch kernel, not
just the small helpers. Both generated PTX call graphs have at most four direct
call edges, twelve reachable functions and no recursion. Ns10 capacity wrappers
also have spill stores/loads of 300/692, 300/440 and 376/728 bytes for capacities
24, 48 and 128 respectively. This reinforces the need for native GPU timing and
occupancy measurement before any efficiency claim. The auto policy remains
one 32-thread block per SM before wall-count/memory caps; explicit slot counts
permit user-controlled experiments. No performance-driven default was retuned
without GPU evidence.

Build logs, source manifests, artifact checksums and dependency receipts were
recorded under `/tmp/wall-gas-final-1f4cf01f-Ns2/` and
`/tmp/wall-gas-final-1f4cf01f-Ns10/`. The first Ns2 attempt stopped during vendor
OpenFOAM environment sourcing before any compiler ran; disabling shell errexit
only while sourcing that environment resolved setup, and strict error handling
was restored before the release build. No compiler flags were weakened.

The final broad comparison log is `/tmp/wall-final-gas-broad.log`; the exact
85-identity comparison is `/tmp/wall-final-gas-failure-delta.json`. The final
workspace test additionally replaces a stale example byte constant with the
real `sizeof` layout; its focused test passes independently of the earlier
whole-repository regression. No README file was modified.


Independent review regression checks: 16 passed across native configuration
lifecycle/storage/rollback and legacy SST lowRe/highRe, mass-only Courant,
internal diffusion and flux consistency. The legacy source-limit check now
uses compile-time optional capability dispatch, so old state types acquire no
new required field. Two source-extraction harnesses were migrated to exclude
the optional wall aggregate and preserve the real production template context;
all original numerical assertions remain unchanged.

### Exact unchanged baseline failure identities

- `ERROR applications/gasUGKP/tests/test_epsg_prev_restoration.py::GasVolumeSourceOrderingContract::test_history_advances_only_after_mass_and_momentum_commit`
- `ERROR applications/gasUGKP/tests/test_epsg_prev_restoration.py::GasVolumeSourceOrderingContract::test_no_particle_and_no_drag_paths_cannot_skip_volume_source`
- `ERROR applications/gasUGKP/tests/test_epsg_prev_restoration.py::GasVolumeSourceOrderingContract::test_volume_source_is_not_applied_twice_on_particle_cells`
- `ERROR applications/gasUGKP/tests/test_gas_solid_coupling_contract.py::GasSolidCouplingContract::test_drag_uses_apparent_gas_capacity_and_intrinsic_writeback`
- `ERROR applications/gasUGKP/tests/test_gas_solid_coupling_contract.py::GasSolidCouplingContract::test_heat_uses_apparent_capacity_and_intrinsic_writeback`
- `ERROR applications/gasUGKP/tests/test_gas_solid_coupling_contract.py::GasSolidCouplingContract::test_volume_history_initialization_is_drag_independent`
- `ERROR applications/gasUGKP/tests/test_gas_solid_coupling_contract.py::GasSolidCouplingContract::test_volume_source_includes_pressure_work`
- `ERROR applications/gasUGKP/tests/test_gas_solid_coupling_contract.py::GasSolidCouplingContract::test_volume_source_is_independent_of_drag_schedule`
- `FAILED applications/gasUGKP/tests/test_collision_sampling_contract.py::CollisionSamplingContract::test_collision_correction_uses_mass_weighted_affine_projection`
- `FAILED applications/gasUGKP/tests/test_collision_sampling_contract.py::CollisionSamplingContract::test_collision_sampling_preserves_heterogeneous_particle_mass`
- `FAILED applications/gasUGKP/tests/test_collision_sampling_contract.py::CollisionSamplingContract::test_collision_sampling_preserves_particle_temperature`
- `FAILED applications/gasUGKP/tests/test_collision_sampling_contract.py::CollisionSamplingContract::test_unresolved_theta_keeps_drag_energy_balance_without_sampling`
- `FAILED applications/gasUGKP/tests/test_cpcst_compaction_dpre_contract.py::CompactionGeneratedDpreContract::test_heavy_preparation_preserves_base_and_injection_segments`
- `FAILED applications/gasUGKP/tests/test_cpcst_compaction_dpre_contract.py::CompactionGeneratedDpreContract::test_injection_tail_honours_warp_aggregated_binning`
- `FAILED applications/gasUGKP/tests/test_cpcst_compaction_dpre_contract.py::CompactionGeneratedDpreContract::test_pre_consumers_traverse_base_and_injection_segments`
- `FAILED applications/gasUGKP/tests/test_cpcst_compaction_dpre_contract.py::CompactionGeneratedDpreContract::test_preparation_bins_only_the_appended_injection_range`
- `FAILED applications/gasUGKP/tests/test_cpcst_warp_reduction_contract.py::CpcstWarpReductionContract::test_component_reduction_uses_a_converged_full_warp`
- `FAILED applications/gasUGKP/tests/test_dynamic_block_contract.py::UGKPBlockConfigurationContract::test_cuda_occupancy_is_the_launch_limit_source`
- `FAILED applications/gasUGKP/tests/test_dynamic_block_contract.py::UGKPHeavyContract::test_dynamic_threshold_and_exact_split_population`
- `FAILED applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_continuum_transport_is_separate_from_riemann_flux`
- `FAILED applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_csr_heavy_and_warp_aggregation_source_contract`
- `FAILED applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_face_kernel_specialization_preserves_laminar_and_les_paths`
- `FAILED applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_muscl_uses_componentwise_bounded_primitive_limiters`
- `FAILED applications/gasUGKP/tests/test_gpu_boundary_schedule_contract.py::GpuBoundaryScheduleContract::test_gpu_uses_clamped_linear_interpolation_for_both_tables`
- `FAILED applications/gasUGKP/tests/test_gravity_drag_contract.py::test_device_drag_model_is_static_and_shared_by_both_consumers`
- `FAILED applications/gasUGKP/tests/test_gravity_drag_contract.py::test_gravity_uses_exact_gas_energy_and_particle_drag_impulse`
- `FAILED applications/gasUGKP/tests/test_gravity_drag_contract.py::test_model_switch_occurs_only_in_host_launch_helpers`
- `FAILED applications/gasUGKP/tests/test_interface_contract.py::UGKPInterfaceContract::test_mu_cp_over_pr_is_the_only_molecular_conductivity_path`
- `FAILED applications/gasUGKP/tests/test_jamming_pressure_contract.py::MobilePackingProjectionSourceContractTests::test_collisional_pressure_kick_no_longer_owns_jamming`
- `FAILED applications/gasUGKP/tests/test_jamming_pressure_contract.py::MobilePackingProjectionSourceContractTests::test_mobile_moment_accumulation_uses_active_particles`
- `FAILED applications/gasUGKP/tests/test_jamming_pressure_contract.py::MobilePackingProjectionSourceContractTests::test_projection_is_cell_iterative_and_particle_single_pass`
- `FAILED applications/gasUGKP/tests/test_jamming_pressure_contract.py::MobilePackingProjectionSourceContractTests::test_projection_keeps_local_support_and_control_counts_on_device`
- `FAILED applications/gasUGKP/tests/test_jamming_pressure_contract.py::MobilePackingProjectionSourceContractTests::test_projection_uses_signed_slack_in_the_lcp_rhs`
- `FAILED applications/gasUGKP/tests/test_no_wall_retention_contract.py::NoWallRetentionContract::test_existing_reflection_coefficients_remain`
- `FAILED applications/gasUGKP/tests/test_openfoam_case_configuration_contract.py::OpenFoamCaseConfigurationContract::test_scheduled_inlet_density_is_derived_from_p_and_t`
- `FAILED applications/gasUGKP/tests/test_particle_cell_path_runtime_contract.py::ParticleCellPathRuntimeContract::test_both_execution_paths_are_present`
- `FAILED applications/gasUGKP/tests/test_population_diagnostics_contract.py::test_injection_count_has_no_per_particle_diagnostic_atomic`
- `FAILED applications/gasUGKP/tests/test_population_diagnostics_contract.py::test_pretransport_count_reuses_an_existing_cell_kernel`
- `FAILED applications/gasUGKP/tests/test_s1_kernel_resource_contract.py::S1KernelResourceContract::test_active_pressure_cache_kernels_are_spill_free`
- `FAILED applications/gasUGKP/tests/test_s1_kernel_resource_contract.py::S1KernelResourceContract::test_l0_full_collision_pool_matches_the_established_resource_budget`
- `FAILED applications/gasUGKP/tests/test_s1_kernel_resource_contract.py::S1KernelResourceContract::test_split_collision_segments_are_spill_free_and_s1_is_lean`
- `FAILED applications/gasUGKP/tests/test_s1_kernel_resource_contract.py::S1KernelResourceContract::test_uncached_drag_and_pool_clear_are_spill_free`
- `FAILED applications/gasUGKP/tests/test_segmented_task_queue_family_contract.py::SegmentedTaskQueueFamilyContract::test_enabled_path_is_not_light_then_heavy`
- `FAILED applications/gasUGKP/tests/test_segmented_task_queue_family_contract.py::SegmentedTaskQueueFamilyContract::test_l2_advance_does_not_launch_dead_light_reductions`
- `FAILED applications/gasUGKP/tests/test_segmented_task_queue_family_contract.py::SegmentedTaskQueueFamilyContract::test_l2_materializes_only_nonempty_tasks_and_fuses_publication`
- `FAILED applications/gasUGKP/tests/test_segmented_task_queue_family_contract.py::SegmentedTaskQueueFamilyContract::test_pool_worker_selects_logical_or_single_source_once_per_task`
- `FAILED applications/gasUGKP/tests/test_segmented_task_queue_family_contract.py::SegmentedTaskQueueFamilyContract::test_split_task_count_uses_one_logical_concatenation`
- `FAILED applications/gasUGKP/tests/test_segmented_task_queue_family_contract.py::SegmentedTaskQueueFamilyContract::test_workers_route_through_the_shared_persistent_protocol`
- `FAILED applications/gasUGKP/tests/test_segmented_task_queue_family_contract.py::SegmentedTaskQueueFamilyContract::test_zero_collision_probability_skips_eight_component_reduction`
- `FAILED applications/gasUGKP/tests/test_serial_translational_cyclic_contract.py::test_backend_maps_periodic_gas_and_particle_transport`
- `FAILED applications/gasUGKP/tests/test_strict_csr_level_contract.py::StrictCsrLevelContract::test_all_examples_use_exactly_one_level_and_no_removed_switches`
- `FAILED applications/gasUGKP/tests/test_tau_wave_contract.py::GasTimeIntegrationSourceContract::test_euler_selector_is_a_single_gas_substage`
- `FAILED applications/gasUGKP/tests/test_tau_wave_contract.py::GasTimeIntegrationSourceContract::test_ssprk2_and_ssprk3_use_gas_conservative_stage_storage`
- `FAILED applications/gasUGKP/tests/test_thermal_segmented_queue_contract.py::SegmentedTaskQueueFamilyContract::test_enabled_path_is_not_light_then_heavy`
- `FAILED applications/gasUGKP/tests/test_thermal_segmented_queue_contract.py::SegmentedTaskQueueFamilyContract::test_l2_advance_does_not_launch_dead_light_reductions`
- `FAILED applications/gasUGKP/tests/test_thermal_segmented_queue_contract.py::SegmentedTaskQueueFamilyContract::test_l2_materializes_one_deterministic_queue_for_every_cell`
- `FAILED applications/gasUGKP/tests/test_thermal_segmented_queue_contract.py::SegmentedTaskQueueFamilyContract::test_l2_occupancy_queries_the_executed_segmented_workers`
- `FAILED applications/gasUGKP/tests/test_thermal_segmented_queue_contract.py::SegmentedTaskQueueFamilyContract::test_pool_worker_selects_logical_or_single_source_once_per_task`
- `FAILED applications/gasUGKP/tests/test_thermal_segmented_queue_contract.py::SegmentedTaskQueueFamilyContract::test_scheduler_protocol_is_shared_between_thermal_workers`
- `FAILED applications/gasUGKP/tests/test_thermal_segmented_queue_contract.py::SegmentedTaskQueueFamilyContract::test_split_task_count_uses_one_logical_concatenation`
- `FAILED applications/gasUGKP/tests/test_unified_scheduling_tools_contract.py::UnifiedSchedulingToolsContract::test_examples_own_scheduling_not_physics_keys`
- `FAILED applications/gasUGKP/tests/test_unified_scheduling_tools_contract.py::UnifiedSchedulingToolsContract::test_periodic_flux_corrections_are_guarded_by_mesh_flag`
- `FAILED applications/gasUGKP/tests/test_unified_scheduling_tools_contract.py::UnifiedSchedulingToolsContract::test_tool_b1_uses_deferred_cuda_event_measurement`
- `FAILED applications/gasUGKP/tests/test_wall_gks_contract.py::RiemannWallSourceContract::test_symmetry_has_no_viscous_or_fourier_transport`
- `FAILED applications/gasUGKP/tests/test_wall_gks_contract.py::RiemannWallSourceContract::test_wall_viscous_shear_and_heat_are_a_separate_additive_path`
- `SUBFAILED(component_limiter_assignment='gasGradientLimiterP') applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_muscl_uses_componentwise_bounded_primitive_limiters`
- `SUBFAILED(component_limiter_assignment='gasGradientLimiterRho') applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_muscl_uses_componentwise_bounded_primitive_limiters`
- `SUBFAILED(component_limiter_assignment='gasGradientLimiterT') applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_muscl_uses_componentwise_bounded_primitive_limiters`
- `SUBFAILED(component_limiter_assignment='gasGradientLimiterUx') applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_muscl_uses_componentwise_bounded_primitive_limiters`
- `SUBFAILED(component_limiter_assignment='gasGradientLimiterUy') applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_muscl_uses_componentwise_bounded_primitive_limiters`
- `SUBFAILED(component_limiter_assignment='gasGradientLimiterUz') applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_muscl_uses_componentwise_bounded_primitive_limiters`
- `SUBFAILED(limiter='pLimiter') applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_muscl_uses_componentwise_bounded_primitive_limiters`
- `SUBFAILED(limiter='rhoLimiter') applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_muscl_uses_componentwise_bounded_primitive_limiters`
- `SUBFAILED(limiter='tLimiter') applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_muscl_uses_componentwise_bounded_primitive_limiters`
- `SUBFAILED(limiter='uxLimiter') applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_muscl_uses_componentwise_bounded_primitive_limiters`
- `SUBFAILED(limiter='uyLimiter') applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_muscl_uses_componentwise_bounded_primitive_limiters`
- `SUBFAILED(limiter='uzLimiter') applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_muscl_uses_componentwise_bounded_primitive_limiters`
- `SUBFAILED(token='1 + (count - 1)/s.csrHeavyTileParticles') applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_csr_heavy_and_warp_aggregation_source_contract`
- `SUBFAILED(token='__match_any_sync') applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_csr_heavy_and_warp_aggregation_source_contract`
- `SUBFAILED(token='countCsrReductionTasksKernel') applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_csr_heavy_and_warp_aggregation_source_contract`
- `SUBFAILED(token='finalizeCsrSegmentedMomentCellsKernel') applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_csr_heavy_and_warp_aggregation_source_contract`
- `SUBFAILED(token='finalizeCsrSegmentedPoolCellsKernel') applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_csr_heavy_and_warp_aggregation_source_contract`
- `SUBFAILED(token='groupMask') applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_csr_heavy_and_warp_aggregation_source_contract`
- `SUBFAILED(token='laneRank') applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_csr_heavy_and_warp_aggregation_source_contract`
- `SUBFAILED(token='materializeCsrReductionTasksKernel') applications/gasUGKP/tests/test_gks_les_contract.py::SourceContractTests::test_csr_heavy_and_warp_aggregation_source_contract`
