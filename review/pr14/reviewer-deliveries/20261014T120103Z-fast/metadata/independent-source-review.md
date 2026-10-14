# Independent source review — PR14 third FAST round

PR14_REVIEW_ONLY. This seat read the fixed source and archived evidence, performed read-only offline analysis, and wrote only these two metadata reports. It executed **zero compiles and zero GPU runs**. No source/index/HEAD changes, merge, PR comment, description change, README edit, new gate, or old OOB rerun occurred.

## Identity and assessment

Base source `4b5c480b572f234ddecf5e89164fe59b0d2e9735`; formal source `9d5be6eff69dbced2fc959d422d3ba419a1bc39e`; formal tree `5dc844d4189f50fcfbed2a77fa5e6cd6e9cfd066`; checkout/record parent `a462edd16c20750a5875548f2e4ec690602d05cc`. Historical GPU evidence stays pinned to `f5e6c5e2cbde2693f5fa811d874c200b07bbbc66` / `review/pr14/reviewer-deliveries/20261014T072053Z-fast/`. The machine-readable report gives every reviewed context blob/hash and reference artifact hash.

**Scoped verdict:** the production volume repair matches the requested zero-mass behavior; no new production defect was found in this diff. The submitted fixture fixes the host/device alias and hardcoded partition problems, and has meaningful accepted-state comparisons. Two P2 test-evidence gaps remain. Source review does not certify whole-PR FAST/GPU acceptance or merge readiness.

## Strengths and exact source pins

The non-review diff contains exactly the following five source/test files; their working bytes match the formal blobs and developer SHA-256 pins.

| Path | Formal blob | SHA-256 |
| --- | --- | --- |
| `applications/CHMT/materials/CpuMaterialDriver.C` | `a728a5daaf0bab41ff33fa954828737c2fce3f01` | `cd79a6a6300bfbf24de7c388a11755270734604bc78053e66a44771b6e345e4e` |
| `tests/fixtures/chemistry_trial/chemistry_trial.cu.in` | `4a7475100f1d9a185603da79de7ceb56e0ef10d6` | `6cf44c609ce59ea79a677a1bda6a3e6ace544fd3052c2d52443ff4fb2981655b` |
| `tests/fixtures/chmt_zero_phase_material/probe.C` | `bef1710a8b5d11ab7469e87bb0259938c6675215` | `96bbcdd9f1ee14f78b8374b9a0a6546b0e2005fdf92f3b9a0ca7e3c2912f1522` |
| `tests/test_chemistry_trial_cuda_fixture.py` | `d281a869aed75bd7597503e4064f651ce7b14eb8` | `5c74cf92e9c6edb51644907ecb1394d8008d22456451972d239038e63c383a2b` |
| `tests/test_chmt_zero_phase_material.py` | `5bd5099eb3d88b3e5312ad8d6519abbea729a770` | `54dbe7c2135ac378b361c57202b7f69fbfe5dc4804e937c7e8ec4c977203ba63` |

## GPU-F006 producer → storage/transfer → consumer → commit

Target file ID `F014-004c176a472f`, `CpuMaterialDriver.C:460–463`, blob `a728a5daaf0bab41ff33fa954828737c2fce3f01`. The conditional is exact zero mass, including signed zero. Its zero arm does not evaluate `condensed[phaseCondensed]`; its nonzero arm retains the original division order and density/porosity expression. It adds no rejection, retry, version/index/precision check, or numerical gate.

`ablation/CpuSurfaceInterface.C:57–71` creates a SolidFilm packet for a wet solid interface even when phase mass is zero, retaining conductive/advective/work energy and pressure/shear momentum. The condensed loop compares the selected integer without indexing it. `film/CpuFilmDriver.C:309–333` finalizes the actual phase packet and checks selection only for nonzero mass in the existing nonterminal path. `CpuMaterialDriver.C:421–432` still finalizes and calls `filmForcing`; `117–122` retains phase mass/species/energy/kinetic-work flags independently of mass. The repair is only at target-volume accumulation. Normal velocity/thickness/geometry at `466–473`, packet batch application and accounting at `548–559`, film candidate advance at `573–587`, packet ledger at `616`, and accepted-state/budget publication at `617–619,670–683` remain on the chain.

Force is preserved as the produced packet and unchanged solid/film/support momentum accounting (`35–40`) and pressure/shear/normal-velocity drive (`123–125,418–420`). `CpuFilmForcing` does not itself contain a momentum member; this review does not claim that budget equality proves complete physical film momentum. GPU inventory observations must establish that separately. Recovering the donor material before the conditional is unchanged.

The old legal unselected heat/evap failures and static OOB remain historical facts. The old exact SIGFPE PC is unresolved; no dangerous old run was repeated. Fresh formal wet/restart/evap/full inventory and signed nonzero controls are required runtime evidence owned by the coordinator.

## Submitted chemistry fixture coverage

`chemistry_trial.cu.in:32–34` uses a heap host owner and a separate `cudaMalloc` device view. Managed arrays remain backing storage; `88` uploads initialized descriptors and `292` re-uploads changed failure inputs. Every direct chemistry launch uses `s->deviceState`. `sync()` (`18`) precedes host observations. Native validation synchronizes and performs real diagnostic copies; shared snapshot copies are device-to-device. Normal cleanup (`312–317`) synchronizes, releases trial/interval snapshots, closes/deletes the host ledger, deletes the heap owner, then frees owned CUDA allocations in reverse order. This removes the earlier managed owner alias; no general allocation/free/sync recovery claim follows from `check()/cudaCheck()` process exits.

The generator (`prepare_chemistry_trial.py:27–49`) includes the native production owner, derives actual scalar-pointer backing fields, and records the nvcc command. The new Python generation tests are source-contract checks, not runtime proof. Default pytest skips are honestly labeled; actual explicit-nvcc compile/link has separate scope.

Accepted comparisons (`104–136`) cover the actual transaction cell/face field lists (`SharedGasTrialPolicy.cuh:38–55`), both cell status values, all face status values, all nine ChemistryAudit members and all six ChemistryStatus members without comparing struct padding. Both transport and chemistry budget arrays are in cellFields; initialTransport is separately compared. Ledger comparison (`137–161`) covers all emitted physical/audit records, removing only `audit_host_seconds`, and compares the listed mutable ledger inventory/time/state members. Direct ledger member comparisons omit deviceBytes, hostReadBytes, identity and path; accepted records cover them indirectly, while the empty direct-chemistry ledger cannot. This is a coverage qualification, not evidence of mutation in the current source.

The requested owner (`201–226`) retains request4e-6, measured real Co, target0.5, genuine native refinement, observed accepted durations, `native=true` replay and exact elapsed==requestedDt. It does not assume two accepted parts. The terminal path (`228–240`) exhausts maximumAttempts1 and compares the complete original interval with public false. The original requested-owner and direct chemistry1e-6 controls remain (`244–266`). Final equality against accepted-part replay makes rollback leakage observable on this owner path, but does not replace detailed failure-case auditing.

## Issues

### Important — P2, SR-R3-001: failure matrix comparison and evidence remain partial

File `tests/fixtures/chemistry_trial/chemistry_trial.cu.in:294–309`; blob `4a7475100f1d9a185603da79de7ceb56e0ef10d6`, SHA-256 `6cf44c609ce59ea79a677a1bda6a3e6ace544fd3052c2d52443ff4fb2981655b`.

The sixteen cases check all four species entries, both complete accepted audit objects and all four active chemistry-budget entries (`287–298`). They do **not** check the complete six-member typed failure status for both cells, all other transaction state/budget arrays, host ledger or a real begin/rollback per configuration. Generic failure and unavailable classification plus selected retry checks admit wrong diagnostic details: e.g. kind13 could change to another numerical failure code with different counters and still pass. Rows7/15 are printed after reset/public execution, so their original failure tuples cannot be recovered from those PASS rows.

This is a test assertion/evidence gap, not a finding that the current production kernel leaks state. The adapter explicitly never writes conservative rho/rhoU/rhoE (`advanceGasChemistryKernel.cuh:76–81`); unavailable/trial failure returns before species/audit/budget publication (`185–191`). Retain the formal original result and use separately identified test-only observations/assertions for all failure semantics and rollback. Do not relabel an augmented support fixture as the submitted original.

Source-derived expected tuples **before public reset/overwrite**, `(code,thermoCode,reaction,attempted,rejected,nonlinear)`, for both identically initialized cells:

| kind | Trigger | Expected tuple | cellStatus | retry / terminal |
| --- | --- | --- | --- | --- |
| 0 | valid 2e-6 interval, initial/maximumStep1e-6, maximumSteps1 | `StepLimit,Success,-1,1,0,effort sum*` | ChemistryFailure | 1 / 1 |
| 1 | thermo.species=null | `InvalidModel,InvalidModel,-1,0,0,0` | InvalidStorage | 0 / 0 |
| 2 | thermo.coefficients=null | `InvalidModel,InvalidModel,-1,0,0,0` | InvalidStorage | 0 / 0 |
| 3 | stoichiometricBasis=null | `InvalidModel,Success,-1,0,0,0` | InvalidStorage | 0 / 0 |
| 4 | reactions=null | `InvalidModel,Success,-1,1,1,1` | InvalidStorage | 0 / 0 |
| 5 | reactants=null | `RateFailure,Success,0,1,1,1` | InvalidStorage | 0 / 0 |
| 6 | products=null | `RateFailure,Success,0,1,1,1` | InvalidStorage | 0 / 0 |
| 7 | coefficientCount=1 with complete supplied backing | `InvalidModel,InvalidModel,-1,0,0,0` | ChemistryFailure | 0 / 1 |
| 8 | zero rank-one basis with backing present | `InvalidModel,Success,-1,0,0,0` | ChemistryFailure | 0 / 1 |
| 9 | reaction preExponential=NaN | `RateFailure,Success,0,1,1,1` | ChemistryFailure | 1 / 1 |
| 10 | reactants=null plus maximumRejectedSteps100/maximumSteps1 | `StepLimit,Success,0,1,1,1` | InvalidStorage | 0 / 0 |
| 11 | ThirdBody with positive efficiencyCount and no efficiencies backing | `RateFailure,Success,0,1,1,1` | InvalidStorage | 0 / 0 |
| 12 | unsupported species model255 | `InvalidModel,InvalidModel,-1,0,0,0` | InvalidStorage | 0 / 0 |
| 13 | maximumNewtonIterations0 | `InvalidControls,Success,-1,0,0,0` | ChemistryFailure | 0 / 1 |
| 14 | stageVolumes=null | `InvalidModel,Success,-1,0,0,0` | InvalidStorage | 0 / 0 |
| 15 | zero interval plus coefficientCount1 and missing basis (basis not consumed) | `InvalidModel,InvalidModel,-1,0,0,0` | ChemistryFailure | 0 / 1 |

*Kind0 nonlinear effort is the sum for one successful full plus two half SDIRK solves; the simple constant mechanism predicts12 from six two-iteration implicit stages, but that inferred count must be recorded from runtime rather than presented as observed evidence. Early-return and one-failed-evaluation counters in other rows follow directly from `LocalStiffIntegrator.H:163–197,213–224`, `ConservativeReactor.H:80–95,109–130`, `GasRates.H:93–110` and the volume adapter (`advanceGasChemistryKernel.cuh:304–305`).

Kind10 correctly returns **StepLimit with reaction0 retained**, not RateFailure code retention. The existing unavailable assertion indirectly catches lost reaction identity via `unavailableTrialModel`; it still does not assert all six semantic fields. Kind15 has thermo metadata failure before basis consumption even at zero interval, so its absent unused basis must not become InvalidStorage. Failed native cells may publish ChemistryStatus and sticky cellStatus; persistent species, accepted audit and chemistry budget remain unchanged.

### Important — P2, SR-R3-002: chronological rejected-attempt trace is missing

File `chemistry_trial.cu.in:169–182`, same exact blob/hash. Aggregate begins/commits/rollbacks/StepLimit counters and accepted dt replay are useful, but validate does not emit the failed dt/native flag and both-cell status tuple, and rollback does not emit restored semantic state. `sourceDurations` is collected only at applySources and not printed, so it cannot capture failed pre-transport chemistry calls. The default-owner assertion proves some StepLimit occurred, not the chronology of the initial rejected call. Record that chain with a separately labeled observer while retaining the complete accepted trace, original control parameters and endpoint.

Rollback semantics must be correct: `SharedGasTrialPolicy.cuh:150–168,263–277,306–313` snapshots before status reset and restores cell/face/status/audit values on a successful rollback. Host `gasTrial.retryableFailure` and `terminalContinuation` deliberately retain the failure decision because `GpuGasAdvance.cuh:781–790` reads them after rollback; do not assert those flags return to the pre-begin false value. gasModelPoisoned stays false without an actual restore error. Trial rollback does not modify host ledger. Interval rollback separately sets requestActive=false (`314–320`); configured missing backing/metadata is input, not rollback output.

No critical finding and no new production bug was established. These issues concern scope and durability of submitted test evidence.

## Host material support and legal signed phase controls

The probe calls the actual complete CPU material driver, including native OpenFOAM conduction and coupled-recession geometry; it does not link a CUDA backend. Its three modes inspect one retained SolidFilm packet, nonzero heat/force, reciprocal energy/x-momentum/support accounting and actual film enthalpy effect. Zero modes check unchanged mass and zero target volume. Positive melting checks the original FP64 density/porosity expression against the actual target (`probe.C:133–143`). Historical selected/noMelting and positive-melting output hex values agree with the safe old source controls, as separately recorded in HOST_RECEIPT/logs. The original unselected bounds-red is retained and was not rerun here. Exact old-control exits and start/end timestamps were not captured; do not invent them.

This is one constructed host owner, one phase direction and normal-path support. It does not prove full native GPU wet/evap/restart budgets, negative solidification, nonzero porosity/pore models, all material physics or error recovery. The developer report states those limits and makes no new GPU claim.

Proposed source-derived native-app controls: create separate clones of the legal old wet geometry (dense C0, initial solid300K, film350K, thickness1e-5, Y=(1 0)); add `material { phaseCondensed 0; enableMelting true; phaseFilmY (1 0); evaporation Disabled; }` and `meshMotion { policy CoupledRecession; }`. Existing tables with equal solid/liquid rho1000 and cp1000, e0_s0/e0_l100000 and liquidReferencePressure171600 give latent99828.4>0. For an independent small1e-6 interval, meltTemperature300 gives positive phase mass; meltTemperature400 gives negative mass. `InterfaceMath.H:295–303,344–363` gives the sign and `725` requires dense initial material. These are proposed inputs, not run guarantees; actual signed mass, donor, porosity/volume, force/heat/film consumers and all budgets must be measured. Never change the original noMelting recheck to these controls.

## Reuse applicability and build limitations

All24 context source dependencies inspected in this chain are byte-identical to the previous source. Independently rehashed both Ns2/Ns10 receipt regions:448 files each match current bytes and source4b5c480. Recorded reused Backend.o hashes are Ns2 `190354b5cb984dcaf53497c72155d22dd734fe64dcf23f4548d944a93f42f447`, Ns10 `2df9a6d20f68f8fe0e833544de6e79d76e89d0efa16003cf5e8b2515274744a6`. Compiler logs/receipts describe incremental host compilation plus final application link; the object bytes under historical /tmp paths were not independently available here. No fresh whole CUDA application TU or runtime acceptance is inferred.

The JSON gives a disposition for all25 historical acceptance units, preserving exact scopes and limits. Unchanged chemical/operator/Co/CHT/FSH/ALE/clock/SST22/radiation/codec evidence can be reused only for its original units. R2-A-owner-full-semantic remains production evidence from S208, not proof of the new submitted fixture. Original fixture FAIL and wet original FAIL, diagnostic-only M/E PASS, evaporation timeout remain immutable; fresh current fixture and changed wet consumer evidence are separate. Dry gas/face/M/E evidence does not prove wet or phase success. Full Ns10 initcheck, active cold, full-vs-restart first difference, momentum/gross SST/full inventories, mapper, cold-start Js and resource/sanitizer gaps are not closed by source identity or compilation. New GPU run count from reuse is zero.

## Strict momentum reanalysis: grounded but scoped

Read and independently called `reanalyze.py::analyze` without invoking its writing main. All three retained RESULT case payloads reproduce exactly. All46 INPUT_HASHES match current bytes; all42 historical paths also match their immutable blobs at f5e6c5. Logger D203 field9 is final flux momentum.x and field11 is paired packet momentum.x. Interface physical gas faces12–15 are correctly selected by BoundaryKind6; persistent packet identities16–19 are a different namespace. The original external `r2_complete_budget.py:14,44` uses the same abs1e-14 + rel1e-10×max(beforePx,afterPx), with binary64 operands converted exactly and80-digit sums.

The replay covers each of100 accepted microsteps/case, all8 cells, all36 final faces/stage, actual RK ledgerDt,24 boundary faces, and8 paired GasSolid/PoreGas channels. Both cumulative boundary/exchange double accumulators match checkpoint bits. Physical final-flux residuals at all300 accepted endpoints and6 macro endpoints meet the original rule. Window2 values:

| Case | Physical final-flux residual Px | Original accumulated-budget residual Px | Original status |
| --- | --- | --- | --- |
| SST-SSPRK2 | +2.8099402275e-20 | +8.0149895103e-12 | FAIL |
| SST-SSPRK3 | +2.0552727120e-19 | +8.0289771413e-12 | FAIL |
| dry-ablation | -3.3259918837e-19 | -3.5785010887e-13 | FAIL |

The arithmetic decomposition is supported by logged execution order and bit replay, not a PASS marker. The5 original cumulative FAIL and dry-window1 PASS remain. This covers three CHMT gas-x-momentum cases only, accepted (not intermediate per-stage) inventories, and inherited observer output-equality comparisons. It is not new GPU evidence, a ledger repair, wet-film recipient/vector/global P, particle/rigid-support accounting, FSH/CHT full budgets, arbitrary forces, or gross SST validation.

## Declined to judge

- **Whole-PR merge readiness and completion of real-GPU FAST acceptance** — This source-review seat runs no GPU or compiles; coordinator owns execution and aggregate assessment.
- **Re-running the known old disabled-index wet/evap input to recover a signal PC** — Explicitly forbidden dangerous OOB rerun; prior static OOB and SIGFPE remain separate from unresolved exact PC.
- **New numerical/index/precision/contract refusal or rollback gates** — Explicitly unauthorized; recommendations apply to test observations only.
- **All melting/solidification/pore/material laws and complete wet/restart/evap inventories** — Only the changed consumer chain and declared host regression were source-reviewed; fresh actual native-application control execution is coordinator work.
- **Active cold full groups, cold-start Js/history-Co, empty/multiple mapper runtime results** — Independent remaining gaps; not touched by this five-file source change and no execution in this seat.
- **Gross SST, complete vector/global/particle/film/support momentum and three-resident full-stage inventory** — Three historical gas-Px offline replays are narrower; original red and incomplete scopes are preserved.
- **General allocation/free/sync/copy/host-allocation recovery and same-owner multithread concurrency** — Fixture check/cudaCheck terminate; normal ownership/synchronization/destruction inspection cannot prove general recovery.
- **Whole fresh application CUDA translation-unit build or independent old /tmp object verification** — Receipts give incremental host compile/final-link plus reused object hashes. Historical object bytes are not available in this checkout; no compile executed here.
- **PRODUCTION long runs, merge, PR comments/descriptions, README changes** — Not authorized; no such action performed.

## Recommendations and scoped assessment

Keep the formal original fixture result and separately label any S301/support observation used to fill the two evidence gaps; production repairs remain developer work. Run the mandatory original wet/evap/restart and both legal signed nonzero controls with unchanged rules, preserving all raw red/blocked results. Reuse only the unchanged units in the dependency table and retain all14 original gap IDs in aggregate records.

The reviewed production expression is appropriate at the pinned source. Whole-PR GPU acceptance and merge readiness are **not established by this source review**, and this task grants no merge authority. The coordinator must make the final FAST assessment from complete actual evidence.
