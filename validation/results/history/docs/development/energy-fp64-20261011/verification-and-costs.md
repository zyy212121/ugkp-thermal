> Historical evidence note (updated 2026-10-11): blocked-migration and old
> coupling conclusions below describe the pinned earlier snapshot, not the current
> paired-carry implementation. The current design/cost contract is
> [paired-carry-design.md](paired-carry-design.md); current verification and
> remaining native-runtime limits are in [current-verification.md](current-verification.md).

# Authoritative-energy implementation evidence and costs

## Historical gas-energy-only scope and status

The snapshot audited below was a local, blocked implementation, not a production-ready CHT precision release. Its unpaired publication and cost statements are preserved as historical evidence. The later paired-carry implementation supersedes this snapshot; use paired-carry-design.md and current-verification.md for its current representation, costs and evidence. Only CHT's two existing rhoE/rhoENext resident arrays are widened. CHMT and gasUGKP already own double energy. The actual-particle publication results in particle-budget.md prevent claiming full gas/particle conservation: a newly retained double gas decrement can have no corresponding float particle temperature/enthalpy publication. Baseline finite-step drag/heat discrepancies are separately characterized for the local actual-kernel fixture; they are not proof of an end-to-end production-advance defect. No new stop or altered exchange model hides either category.

The independently committed EOS/RK/zero-time fixes repair valid-state identity without requiring the storage migration. The FP64 golden change is narrowly audited in double-eos-audit.md. No README file is changed.

## Regression evidence

- Real32 old actual divergence loses each 0.02, 0.03 and 0.05 J increment against 1e9 J. The mixed authority regression executes the actual production kernel with its final stored flux, checks an independent high-precision reference, and tests signed/tiny increments and 10,000 updates. Limits were specified before implementation: at most two double stock ULPs per update and 0.2% relative physical increment error for increments at least 0.02 J.
- Actual source probes cover gravity work, phase-volume source, gas-side drag internal transfer, gas-side finite-capacity heat decrement and identity source updates. Those gas-side tests alone do not establish paired publication; the separate actual-particle suite records the published momentum, unresolved energy and material temperature/enthalpy.
- EOS coverage includes mixed float tables/double scratch, negative formation energy, high kinetic/small internal energy, Ns2/Ns10 chemistry, valid-state bit identity, true invalid/lower-bound repairs, representability of derived float pressure/temperature/sound caches, and failure nonpublication. Old homogeneous float gravity's time arithmetic is explicitly retained and tested.
- Recovery/chemistry consume authoritative double energy directly; chemistry has no additional reaction-heat write. Real species publication and table coefficients retain their existing types.
- Shared binder/POD, RK saved authority, invalid-ALE nonpublication/restore, explicit Vold/Vnew and Euler-only gating are exercised. CHT has no general rejected gas-step transaction or ALE owner; its actual fixed-grid OpenFOAM restart lifecycle is documented in restart-audit.md.
- Exact baseline zero-time drag probe: Real32 fails with changed conserved bits, Real64 passes. The focused two-line early return makes both pass, while coupling caches and thetaDragAlpha are still refreshed.

## Persistent memory and passes

For N CHT cells in Real32, rhoE adds 4N bytes and rhoENext adds 4N bytes: exactly **8N additional persistent device bytes**. No additional energy array is allocated. Existing host input/output mirrors were already double. The pointer-only POD view has no size increase. Existing double owners add zero persistent bytes. No new whole-grid conversion or energy reconciliation pass is added. At this historical gas-only checkpoint, the pairing repair had not been implemented and was not hidden in these costs. The later paired implementation has additional allocations and traversals, explicitly itemized in paired-carry-design.md.

Mixture/chemistry scratch uses double with native table storage and no per-cell table copies. Its local-memory cost is substantial, as measured below.

## Native compile scope

CUDA 13.1, sm_89, C++17, O3, --fmad=false and ptxas verbose were used for CHT32/64 and the six chemistry instantiations. They compiled without requiring a GPU. Existing unused-variable and constexpr host/device warnings remain. The exact original CHT32 base initially failed with 41 namespace/include-order errors; the isolated dependency include fix alone made that base compile.

The following CHT measurements are from the **earlier working snapshot**, before the final EOS momentum/representability and zero-exchange refinements. They are not final-commit compile or performance evidence; a final serialized native build remains required. Isolated output directories prevent replacing another worker's executable. CHMT's attempted standard O2 Ns2 compile ended with exit 137; neither OOM nor a source defect is proven by that result. Heavy compilation was then paused by the parent.

### Earlier CHT register counts (base to migrated snapshot)

| Kernel | Real32 | Real64 |
|---|---:|---:|
| RK blend | 20 to 22 | 20 to 20 |
| RK save | 18 to 18 | 18 to 18 |
| Divergence | 36 to 37 | 40 to 40 |
| Schiller-Naumann drag | 42 to 46 | 68 to 68 |
| Gidaspow drag | 42 to 46 | 70 to 70 |
| Shared primitive recovery | 29 to 42 | 44 to 44 |
| CHT primitive wrapper | 28 to 42 | 46 to 46 |
| Volume-fraction source | 36 to 44 | 50 to 52 |
| Gravity | 26 to 35 | 34 to 34 |
| Material heat | 36 to 40 | 60 to 60 |

All listed CHT entries reported zero stack frame and zero spill-store/load bytes in that snapshot.

### Actual chemistry instantiation costs

These are compiler reports per kernel/thread, not occupancy or runtime estimates. State/Table denotes storage precision; Energy denotes local authoritative scratch precision.

| Ns | State/Table | Energy | Registers | Stack bytes | Spill stores/loads bytes |
|---|---|---|---:|---:|---:|
| 2 | float | float | 255 | 752 | 28 / 72 |
| 2 | float | double | 255 | 1856 | 40 / 104 |
| 2 | double | double | 255 | 1904 | 36 / 80 |
| 10 | float | float | 255 | 8800 | 32 / 104 |
| 10 | float | double | 255 | 16512 | 40 / 168 |
| 10 | double | double | 255 | 17120 | 36 / 144 |

No GPU event timing, full-stage timing, occupancy measurement or memcheck was possible. These compile costs do not justify a speed or runtime-safety claim.

### Preserved artifact hashes

SHA256 for isolated earlier-snapshot compile artifacts:

- Base32: 0c3611158cd5bd74c4c0fc6f6e64026e1c5fba8428fcd7dea8a11503ed848cb1
- Base64: 784465fb00948779a9deb52628d60d0630c0e30a07ab489333088d2f4c5ed0db
- Earlier migrated CHT32: bab60593333b65166d5551802b736e2f57a607ea04065423d50447b42662daf4
- Earlier migrated CHT64: 67ba8e03269ef867aea7678caf1e8f92fd55b0b4448c9fc735d4fcaeb1f864bf
- Chemistry six-instantiation object: 86590fb9254c85922ff56c4104412bc127abc151ab43cba394c826e0427f5215

The CHT working snapshot was not frozen as a commit when these objects were built; these hashes must not be associated with the final source revision. Final integration needs a new exact source/object mapping.

## Independent narrow-fix verification

An isolated `git archive 197302f` (without the uncommitted precision migration)
passed 13 tests: double-energy recovery, zero-time drag and original gas-state
view/golden coverage. A subsequent restart-test cleanup separates its existing
double copy-helper contract from the pending owner-type promotion; the owner
assertion is retained in the migration suite. With that test-only cleanup,
original-owner restart and global-namespace dependency probes passed all four
checks. Independent read-only review found no blockers in the momentum recovery
and zero-time commits.

The current working implementation separately passed 67 focused energy/source/
actual-particle/zero-time tests, including both exact-baseline publication
probes. One additional owner live/RK double-type check passed. Characterization
success still does not accept the documented paired-energy deficit. CHMT's
separate HOST suite passed all 68 tests, and the focused CHT restart suite passed
all three. Managed inventory checks covered 305 standalone mirrors (no peer
checkout comparison); typed field lifecycle probes passed CHT32/64, gasUGKP64
and FSH64 closure.

A prior unrestricted repository-root pytest invocation had 38 duplicate-module
collection errors from existing same-name files in separate test roots. The
project roots are therefore run separately. One intermediate `tests/` run passed
893 tests with 201 skips and 14 setup errors because it collected the particle
fixture while that test was still being edited (duplicate nonFiniteDevice).
That fixture was corrected and independently rerun successfully. Final frozen
collection results are recorded below rather than presenting that intermediate
run as a full pass.

Final frozen `tests/` run with isolated dependencies and the OpenFOAM 10 CPU environment: **910 passed, 201 skipped, 11 subtests passed**, zero failures/errors, in 366.11 s. Skips are retained environment/feature gates, not all labeled CUDA. The subsequently separated owner-type assertion passed independently (one test). No production source changed during this frozen run. The optional exact baseline particle probes ran in the 67-test focused invocation above.
