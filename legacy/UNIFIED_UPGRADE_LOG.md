# UGKPUnified upgrade log

## 2026-09-04: GPU-resident scheduling-control completion (completed)

### Scope and invariants

- The change is applied directly to `UGKPUnified_clean` on its current `main` checkout. No isolated branch or secondary worktree is used.
- `gasUGKP`, `FSH`, and `CHT` are updated together.
- Governing equations, numerical fluxes, particle models, random-number state, physical parameters, iteration counts, convergence criteria, and case dictionaries are unchanged.
- No production case is stopped, attached to, modified, or launched by this work item.

### Baseline state

Baseline commit: `4b34dde` (`Keep particle scheduling control GPU resident`).

The baseline had already completed these scheduling changes:

1. Dpre/Dpost dynamic L2 task thresholds are calculated on the GPU instead of downloading each directory population every step.
2. CHT and FSH do not download split-Dpre base and injection populations during normal directory construction.
3. Split-Dpre/Dpost phase transitions and post-compaction pointer publication do not upload the complete `DeviceState` every step.
4. ToolB3 publishes the automatic L2 decision with a device kernel. Its scalar diagnostic downloads occur only at the configured ToolB3 interval.

One control path remained incomplete in all three solvers. Mobile packing projection downloaded the seed count every enabled particle step. When seeds existed, it additionally downloaded the pressure-active and correction-active cell counts. These synchronous scalar reads forced the CPU to wait for preceding GPU work.

### Implementation plan

1. Preserve the five existing preparation stages: clear packing moments, clear activity counts, accumulate particle moments, normalise cell moments, and prepare projection seeds.
2. Move all count-dependent stages into one cooperative GPU kernel shared by the three solvers.
3. Read and validate the seed, pressure-active, frontier, and correction-active counts only on the GPU.
4. Preserve the zero-seed fast path inside the cooperative kernel so an inactive step returns immediately instead of launching every projection stage over empty lists.
5. For a nonzero seed count, preserve the frontier depth, active masks/lists, one-cell correction halo, projected-Jacobi iteration count and buffer swaps, conservative face correction, cell reconstruction, and one particle writeback pass.
6. Query cooperative-launch support and the actual cooperative-kernel occupancy once at solver creation. Launch no more blocks than can be simultaneously resident, making every grid-wide barrier valid.
7. Keep solver-specific particle eligibility outside the common implementation: CHT and FSH exclude stuck particles; gasUGKP retains its existing active-particle rule.
8. Delete the superseded host-controlled projection kernels and all three scalar D2H branches.
9. Update the three source contracts, build all solvers, and run source tests without starting a case.

### Rejected intermediate design

A fixed-cell-grid version was compiled during development but was not retained. It removed the host reads while still launching every frontier and Jacobi stage when the seed count was zero. With the default 20 projection iterations, that would replace one scalar synchronization with roughly sixty empty kernel launches on the common zero-seed path. The final implementation uses one cooperative launch and performs the zero-seed decision on the device.

### Completed implementation

- `common/gasNumerics/GpuPackingProjectionCooperative.cuh` is the single shared implementation of all count-dependent projection stages.
- `completeMobilePackingProjectionCooperativeKernel()` owns the GPU-side seed fast path and the complete nonzero-seed projection sequence.
- `checkedMobilePackingCount()` traps corrupt counts outside `[0,nCells]` before a list is consumed.
- Grid-wide barriers preserve the former inter-kernel stage ordering and global-memory visibility.
- Cooperative block count is derived once from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`; unsupported cooperative launch fails explicitly during solver creation when mobile packing is enabled.
- `applyMobilePackingProjection()` contains no seed, pressure-active, or correction-active host variable and no D2H transfer.
- The obsolete per-stage global kernels and host-side launch sequence were removed.
- CHT/FSH and gasUGKP retain their previous, distinct stuck-particle eligibility semantics through local `mobilePackingParticleEligible()` functions.
- The completion status replaces the earlier planning status because implementation, source contracts, and compilation are complete.

### Acceptance gates

- No `copyToHost` or `cudaMemcpyDeviceToHost` occurs inside any `applyMobilePackingProjection()`.
- A zero-seed step executes one count-dependent cooperative kernel and exits on the GPU; it does not execute empty frontier/Jacobi/reconstruction stages.
- A nonzero-seed step has the same mathematical stages and configured iteration count as the baseline.
- The common cooperative kernel is occupancy-limited to a legal resident grid.
- Source contracts pass for all three solvers and top-level `Allwmake` succeeds.
- End-to-end speedup, power increase, and exact production timing are not claimed without a controlled GPU A/B measurement.

### Verification evidence

- Targeted mobile-packing source contracts: `gasUGKP 19 passed`, `FSH 20 passed`, `CHT 18 passed`.
- The common cooperative kernel compiles for `sm_89` with 72 registers per thread, no local memory, and no dynamic shared memory in gasUGKP and CHT binaries.
- `git diff --check` passes.
- Static inspection confirms that the three host projection functions contain no count download and that the cooperative kernel contains the device-side zero-seed return.
- CHT full source-contract suite: `140 passed, 21 skipped`.
- FSH full source-contract suite: `219 passed, 6 skipped`. Its postprocessing contract now resolves the permanent case asset path rather than the disposable results directory.
- gasUGKP full suite: `239 passed, 11 skipped, 4 pre-existing failures`. The unchanged failures reference two removed legacy drag helper names, demand an unsafe one-barrier form from the shared segmented-pool worker, and use a multiline scheduling regex without multiline mode. Targeted contracts for this change pass.
- Top-level `Allwmake` built all three solvers without a new compiler warning. Solver source hash: `f010084ff323e57454c6ce0639a4aaed64b9595d3a20576840d8e420f691971f`.
- Final build logs:
  - `applications/gasUGKP/private_backend/build_logs/ugkp_weighted_parcel/separated_build_20260904_111850.log`
  - `applications/FSH/private_backend/build_logs/fsh_finite_contact/separated_build_20260904_111930.log`
  - `applications/CHT/docs/build_logs/ugkpcht_upgrade/cuda_solver_build_20260904_112016.log`
- Final executable SHA-256 values:
  - `gasUGKP`: `dc42523690b05ff17a000fe2edb60b658fd3483c74e04bed4efbd8e5044308b0`
  - `gasUGKPCudaBackend`: `ac4b1542cab706ca72f758524f77a7a413f07535aebdbb0375b88068ae7912c5`
  - `FSH`: `fc93860b522337960b39cf6cde297b1083689bc99db9dd0be250c282cefadd3a`
  - `FSHCudaBackend`: `ad07a715c984f2a04fd88210314e3cabc75a75abb3360a3d83f7fdb2e64be618`
  - `CHT`: `8337bcb0254b7f8a8325bae53c69a48ab03b9bf8ae9b31ffa055dd19ac8efc10`

### Explicitly unperformed verification

- No case or long GPU test is launched because the available RTX 4060 is occupied by an unrelated production process.
- No production-server process is touched.
- ToolB3 interval diagnostics, configured Courant updates, CHT/radiation coupling, and write-time persistence remain intentional bounded host interactions; this item removes the leftover per-step mobile-packing control synchronization.

## 2026-09-04: opt-in CHT wall-contact failure diagnosis

### Trigger and diagnosis plan

- The server dense/L2 case repeatedly passes the first checkpoint after restart and then fails before the next checkpoint. The development sparse/L0 case on the RTX 4060 does not fail.
- `CUDA_LAUNCH_BLOCKING=1` located the first reported CUDA failure in `accumulateParticleWallRepresentedContactAreaKernel`; the later mobile-packing seed-count error was only the next synchronization point.
- The local machine does not contain the server dense checkpoint and therefore cannot reproduce that state quickly. The selected route is an externally enabled diagnostic kernel that records every state value needed to distinguish all deliberate traps in the production contact-area kernel.
- The production kernel, its launch, its arithmetic, and all physical models remain unchanged when diagnostics are disabled.

### Completed implementation

- `UGKP_WALL_CONTACT_DIAGNOSTICS=1` selects a dedicated CHT diagnostic contact-area kernel; absence of the variable preserves the original production kernel.
- The diagnostic path records only the first failure and reports: failure code, directory entry/count, particle array index and original ID, active/stuck state, face and candidate type, deposition area, duration, maximum area, peak fraction, contact age, frozen/kinematic/physical/represented areas, diameter, parcel mass, solid density, and particle temperature.
- Failure codes cover invalid directory particles, invalid/noncandidate faces, nonpositive deposited area, invalid transient metadata, unknown wall state, and invalid represented area.
- The diagnostic path replaces traps with a device error record and performs one diagnostic D2H read per contact-area pass. This overhead exists only while the external switch is enabled.
- Default production execution performs no diagnostic allocation, no diagnostic D2H copy, and no diagnostic per-particle checks.
- The change is confined to CHT because the observed failure is in the CHT-only wall-contact area scaling path used by the server case.

### Verification

- The project CHT direct-link build completed successfully with CUDA `sm_89`.
- `git diff --check` passes.
- No case was launched and no production process was touched.

### Rollback

- The diagnostic change is isolated to `applications/CHT/gpu/GpuResidentStrict.cu` and this log entry. It can be reverted as one commit after the server failure record is collected, or retained disabled with the production path unchanged.
