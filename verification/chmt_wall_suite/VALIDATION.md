# Reproducible CHMT / gas wall verification suite

This test-only suite generates fresh real solver inputs, invokes production executables, and reads their actual results. No solver output is fabricated. No test modifies a production source file. `suite_case.json` is an input manifest, not evidence of execution.

## Evidence and exit status

- `PASS`: a requested executable ran and satisfied a declared hard numerical gate. CPU kernel probes identify their host shim; they are not whole-solver or CUDA validation.
- `FAIL`: compilation, execution, input import, mesh quality, finite-value checks or a hard numerical gate failed. Failed nonlinear convergence is retained.
- `NOT_RUN`: the requested executable/device/output was absent. Exit 3, including `--required`; never counted as passing.
- `UNSUPPORTED`: the requested pairing/backend is outside the implemented scope. For example, CHMT ordinary `wallFunction` and the CPU geometry builder as a CUDA kernel.
- `NOT_APPLICABLE` (check scope only): legacy single-gas inputs do not have shared-mixture metadata; this does not make their native solver case unsupported.
- `REPORT_ONLY`: an actual category 2/3 native run completed; its discrepancies are reported without an agreement gate. Preparation or mesh checking cannot produce this state.

`run_status.json` separates mesh, import, native execution, CPU reference evaluation and validation. It records input SHA256, source commit, frontend SHA256, separately resolved gas-backend path and pre/post SHA256, executed commands and per-command monotonic process wall time (including startup/I/O, not GPU kernel timing), solver build metadata where available, and CUDA availability. `metrics.json` records targets, actual values, L1/L2/Linf, conservation diagnostics and thresholds. `suite_case.json` additionally records grid parameters, time controls, species count, equation/model choices and material cards. A reference computation by itself is not a passed solver test.

## Quick start

Use a Python environment with NumPy, SciPy and pytest. Source the actual OpenFOAM environment before running mesh commands. Native execution additionally needs a compatible GPU, CUDA runtime and built species-count-matched production solvers.

```sh
S=verification/chmt_wall_suite
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider "$S/tests" -q
python "$S/suite.py" --output /tmp/chmt-mesh --mode mesh --required
# Real native execution. Every invocation generates fresh input directories.
python "$S/suite.py" --output /tmp/chmt-native --mode execute --required \
  --gas-solver2 /absolute/path/Ns2/gasUGKP \
  --gas-solver10 /absolute/path/Ns10/gasUGKP --chmt-solver /absolute/path/CHMT
```

The full matrix selects the Ns=10 reactor binary separately from Ns=2 transport cases. The reactor can also be generated and run on its own:

```sh
python "$S/generate.py" gas_chemistry_reactor --output /tmp/reactor10
python "$S/run.py" /tmp/reactor10 --mode execute --required --solver /absolute/path/Ns10/gasUGKP
```

For gasUGKP, the backend defaults to `gasUGKPCudaBackend` beside the resolved frontend executable. Keep each species build in its own directory. An explicitly set `GAS_UGKP_CUDA_BACKEND` selects that exact file; relative paths, including bare names, resolve against the generated case directory, with no PATH search and no fallback. The runner resolves and hashes the selected backend, pins its absolute path in the solver environment, and checks its hash after execution. A missing backend gives `NOT_RUN`; a changed backend fails provenance. Frontend-only hashes do not identify the CUDA implementation.

Every gas case uses `writeControl adjustableRunTime` and 17-digit time/output precision. Adaptive steps therefore align with the output schedule; the ten-species reactor retains its pinned write interval and step size. Completion requires the actual subprocess return code 0, final `simulationTime` at the requested endpoint, one unambiguous exact final-time directory, and finite, correctly shaped rho/p/T/U/rhoE (including solver-created rhoE), and all declared initial species fields. Shared-mixture runs additionally require `End` and the matching species count/mode. Legacy resident gas runs do not emit `End`, so their final progress plus exit/field evidence is the explicit completion contract. Nearby pre-end outputs and an `End` marker alone cannot satisfy it.

The single/legacy gas cases use their original gasUGKP entry. Mixture cases require production mixture mode and the declared species count in the actual log. CHMT requires the native build manifest. There is no automatic CPU fallback.

### CPU production-kernel checks and native CUDA kernel entry

```sh
python "$S/host_cases.py" --backend cpu --output /tmp/chmt-host --required
# Real CUDA kernels, not host emulation. CUDA absence gives NOT_RUN/exit 3.
python "$S/host_cases.py" --backend cuda --output /tmp/chmt-cuda --required \
  --cases wall_constant_limit wall_finite_rate_bvp wall_ten_species wall_sst_robustness
# Compile/link-only evidence, explicitly NOT_RUN even when compilation succeeds:
python "$S/host_cases.py" --backend cuda-build --output /tmp/chmt-cuda-build \
  --cases wall_constant_limit wall_finite_rate_bvp wall_ten_species wall_sst_robustness
```

The CUDA probe wrapper runs the same production wall headers in a real `__global__` launch, synchronizes, checks CUDA errors and copies its return code. Its default compilation architecture is sm_89; change the explicitly versioned build argument for another target architecture. CPU CHMT boundary probes reuse the shipped allocation shim and call actual production boundary/mirror/flux functions. The geometry builder is a host utility. These two adapters have no CUDA substitution; the whole-solver suite is the separate native integration entry. `--source-root` can identify an explicitly chosen production worktree and records its revision and dirty status.

### Real input import, distinct from physics execution

```sh
g++ -std=c++17 -O2 -DUGKWP_GAS_SPECIES=2 -I. \
  "$S/gas_model_probe.cpp" -o /tmp/gas-model-input-check2
# Repeat with UGKWP_GAS_SPECIES=10 for the pinned chemistry reactor.
python "$S/run.py" /tmp/chmt-mesh/gas_reacting_wave --mode check-input \
  --input-checker /tmp/gas-model-input-check2 --required
# CHMT frontend built with its documented --frontend-only build route:
python "$S/run.py" /tmp/chmt-mesh/chmt_receding_slab --mode check-input \
  --input-checker /absolute/path/CHMT-input-check --required
```

The gas metadata checker calls the production thermo/mechanism parser, but does not import the full OpenFOAM frontend or execute a wall closure. CHMT's checker really imports its OpenFOAM mesh/fields/configuration. Record these different scopes accurately.

## Category 1: hard numerical gates

Limits are committed constants. Do not alter them to make a candidate pass. The production wall operator uses binary64 internally in both global binary32 and binary64 builds: nonlinear state, residual/Jacobian, thermo/rates and the wall linear solve are not downgraded with the global state type. In a binary32 build, borrowed thermo/mechanism coefficients remain the actual quantized float tables, values are promoted for arithmetic, and published wall outputs are explicitly converted back to float. Promotion cannot restore information already rounded in those tables.

Old generic-float experiments with a 0.01 nonlinear tolerance are not an acceptance contract for this binary64 internal operator. Keep the declared internal convergence and physical-error gates; assess the actual converted outputs and their recomputed species/budget residuals rather than only the pre-conversion solution. The production precision tests cover default-tolerance float-state solves, finite/transactional writeback and independent long-double NASA/equilibrium/Troe references using the stored coefficient values. This suite's standard CPU wall probes are separately labelled host/double checks. Passing them or a host coupled-Couette reproduction does not establish a native GPU case pass; the N128 native Couette gate still requires its own completed GPU fields and final wall diagnostics.

| Case | Independent reference | Hard targets / normalization |
|---|---|---|
| `small_couette` | Exact transient sine series, finite-volume averages; H=1 m, Uwall=1 m/s, nu=0.1 m2/s, t=0.05 s | U-normalized L2 <=0.01, Linf <=0.025. Not the incorrect steady linear profile. |
| `small_sod` | Exact Euler Riemann solution; gamma=1.4, left (rho,u,p)=(1,0,100000), right=(0.125,0,10000), interface=0.5 m, t=0.0007 s | L1 rho/1, u/sqrt(140000), p/100000 each <=0.03. Conservative cell-average conversion uses 64-point quadrature; report all three norms and global balances. |
| `small_cht_contact` | Initial dry two-region resistance: UA=2 W/K, deltaT=300 K =>600 W | First accepted material energy increment/dt within 5%; closed formation-inclusive total energy residual <=2e-10. |
| `wall_constant_transport` | Canonical N128, Kurganov/full MUSCL; actual native laminar frozen wall closure coupled to transient Couette | U-normalized L2 <=0.02, Linf <=0.05. Initial wall transient vs matching distance is a declared limitation, not a steady exactness claim. |
| CPU `boundary_slip_outlet` | Slip: lambda A (600-300)/d=300 W, zero penetration/shear. Outlet: prescribed 90000 Pa survives both refreshes | Heat absolute error <=1e-9 W; mass <=1e-13 kg/s; shear <=1e-12 N; pressure <=1e-9 Pa. Flux response to changed pressure is a sensitivity check, not an independent exact Riemann-flux oracle. Both single and mixture modes. |
| `wall_constant_limit` | Analytic conduction/Couette/Stefan blowing | tau=0.02 Pa; q=-800.1 W/m2 including viscous heating; blowing q=-m cp deltaT/expm1(m cp L/lambda). Relative Linf <=1e-10. |
| `wall_finite_rate_bvp` | Independent SciPy collocation of first-order exothermic A->B, differential mass-corrected diffusion and formation-inclusive energy | Fixed 0.5% normalized Linf objective for wall Y, q, matching Js and profiles, hard at N96. N24/N48/N96 refinement retained; N24 default is not a universal accuracy claim. Requires production parser/capacity support through N128. Species balance <=1e-9 kg/m2/s; formation-inclusive flux constancy <=1e-7 relative. |
| `wall_ten_species` | Equal-thermo diffusion/reaction: Y0(0)=1/cosh(sqrt(k/D)L), Y9=1-Y0; eight inert species stay exactly zero | Species Linf <=0.006; sum error <=1e-12; inert <=1e-12; isothermal q <=1e-6 W/m2; species balance <=1e-9 kg/m2/s. N48. |
| `wall_polyhedral_geometry` | Exact cube, skew-prism and tetrahedral volume/first/second moments | Absolute component errors <=1e-10 in stated SI moments. Matching is an explicitly zeroth-order containing-cell sample, not affine-exact interpolation. |
| `wall_profile_quadrature` | Exact polynomial antiderivatives for profile-aligned source integration in cube/skew-prism/tetra at N24/48/128 | Normalized absolute error <=2e-11, scale 1+abs(exact); positive weights, volume/xyz moments and linear-in-N storage bound checked. CPU geometric preprocessing only. |
| `wall_sst_robustness` | Declared admissible matching state U=30m/s,T=500K,k=0.5,omega=200, mass flux 0/1e-9/0.001kg/m2/s | All three must converge within N32/100 iterations. A FAIL is retained; this test alone makes no SST accuracy claim. |

Shared-mixture wave, fixed flatplate and MSS7 inputs explicitly use Kurganov/full MUSCL, rather than inheriting the template's unsupported HLLC. The wall Couette gate also uses Kurganov/full MUSCL at exactly N128; its original 0.02/0.05 limits remain unchanged. First-order fallback, mesh refinement, and reference-only tests cannot replace that gate. The reactor retains its separately declared Tadmor/first-order reference fixture. Real OpenFOAM parser/capability and Time-scheduler tests cover these generated controls; these host-side checks do not demonstrate GPU evolution or accuracy.

The BVP equations, boundary conditions, constants, reference residual, all actual node samples and N24/48/96 errors are emitted. No production rate or wall solver is used to calculate the independent BVP reference. For the energy diagnostic, the sampled production profile is independently reconstructed with the declared Fick and conductive constitutive laws and compared to the BVP's constant total enthalpy flux.

## Category 2: transport, chemistry and moving material, report only

- `gas_species_wave`: periodic two-species exact advection-diffusion wave, equal molecular/caloric properties. Reads both actual Y fields; reports their norms, sum, positivity, mean species mass, bulk rho/p/T/rhoE errors and mass/energy residuals.
- `gas_reacting_wave`: same spatial transport plus genuine production A->B chemistry at 2/s. Exact YA is the whole frozen wave multiplied by exp(-2t); YB=1-YA. This exercises simultaneous transport and reaction.
- `gas_chemistry_reactor`: actual native ten-species spatially homogeneous fixture against the existing pinned Cantera 3.1.0 reactor history. Useful chemistry reference, not spatial transport evidence.
- `chmt_receding_slab`: material surface reaction C0->S0, species transport and CoupledRecession, frozen gas chemistry, laminar lowRe.
- `chmt_reacting_receding_slab`: additionally genuine S0->S1 gas reaction at 2/s, laminar lowRe.
- `chmt_laminar_reacting_wall`: same controlled coupled mechanism and motion with `boundaryLayer/finiteRate`, SST disabled. This is a synthetic verification case, not empirical ablator validation.

Every CHMT case emits the complete `material_card.json`, including thermo, kinetics, emitted gas names/parameters, synthetic elemental balance, applicability and source. `controlled_v1` and `controlled_v2` have different density, heat capacity, energy offsets, conductivity and conversion rate, using the same equations and field schema. Select with `{"material":"controlled_v2"}`. `MATERIAL_SCOPE.md` states supported laws and unsupported real-material mappings. Energy includes formation offsets; a separate latent heat source is not double-counted.

Closed cases report total mass/energy inventory changes. Material volume consistency uses every named condensed component's actual density and each cell's saved porosity: sum(mass/rho)/(1-porosity), compared between endpoints. Cell identities, density metadata, finite nonnegative masses and porosity in [0,1) are checked. This endpoint consistency can be partly tautological when porosity is recovered from volume; it is explicitly not an independently integrated surface-recession ledger. The separate controlled constant-rate history remains the independent surface reference. Open-flow cases label inventory changes honestly; without boundary-integrated fluxes these are not conservation residuals. No category 2 target is a hard agreement gate.

## Category 3: real paired wall-treatment levels

`flatplate_fixed`, `mss7_fixed`: gasUGKP SST with `lowRe`, `wallFunction`, `boundaryLayer/reactingSst` variants. `flatplate_moving`, `mss7_moving`: CHMT moving material with `lowRe` and `boundaryLayer/reactingSst`; ordinary CHMT `wallFunction` has an explicit UNSUPPORTED record. `constantTransport` is a laminar analytic limit, not an SST wall-family alternative.

The production fixed ordinary `wallFunction` capability now supports fixed, impermeable `mixtureFrozen` + SST, using local mixture transport and thermodynamics (constant molecular Prandtl or direct conductivity). It does not switch to legacy single-gas physics. Reacting mixtures, ALE, particle coupling, and CHMT ordinary `wallFunction` remain outside that capability. Three-way fixed-wall agreement still requires actual native runs; a passed frontend capability test is not that evidence.

Here “levels” means these near-wall treatments. Pure-gas cases do not invent particle L0/L1/L2 dispatch controls and contain no particles. The inputs use the genuine gasUGKP location `constant/fluidProperties` -> `turbulence`; a standalone `momentumTransport` file is not read by this frontend. CHMT uses `chmtProperties/sst`. `finiteRate` and `reactingSst` share the finite-rate core, with the explicit SST enable flag distinguishing laminar/SST equations.

The user selects meshes/y+. No single prescribed y+ is represented as achieved. Example option file for `suite.py --options options.json`:

```json
{
  "flatplate_fixed": {"nx": 192, "height": 0.2, "end_time": 0.2},
  "wall_families": {
    "lowRe": {"ny": 128, "grading": 80},
    "wallFunction": {"ny": 40, "grading": 5},
    "boundaryLayer_reactingSst": {"ny": 48, "grading": 8, "wall_nodes": 48}
  }
}
```

Those numbers are example mesh controls, not validated y+ targets. Use measured tangential traction, trace density, molecular viscosity and actual owner distance to calculate y+. The new gas CSV provides those; missing lowRe fluxes remain UNAVAILABLE. No coarse-cell gradient is substituted for a modeled wall flux. Stage time is kept distinct from write time. Without face area, q remains W/m2 rather than being mislabeled integrated W.

For this suite's resolved-lowRe baseline eligibility only, new class-3 manifests declare maximum measured y+ 1 in the final 10% accepted-stage window. The wall CSV must belong to the requested final output. A coarse run near y+ 51 is INELIGIBLE; absent y+, absent predeclared targets, stale output or an out-of-window stage is UNAVAILABLE. Meeting the target gives ELIGIBLE_BY_MEASURED_YPLUS for the exported measurements, while physical accuracy remains NOT_ESTABLISHED and pair comparison remains REPORT_ONLY. This is neither a universal turbulence boundary-admissibility rule nor a claim of full-trajectory coverage. The suite does not adjust meshes or relax these targets.

MSS7 uses the existing physical 3D sector points/faces with lateral Slip faces. The near-tip tetrahedron slicing cancellation is repaired in a local coordinate frame with nearer-endpoint interpolation; the original geometry and certification tolerances are unchanged. The real OpenFOAM frontend can certify all 164 default physical wall faces. This is a geometry/import result, not an evolved-flow or experimental-reference result. It does not implement wedge/axisymmetric equations or claim equivalence to the original experiment. The existing digitized-temperature CSV is an unresolved LFS pointer and is not usable reference data. To change near-wall resolution, supply `gas_mesh` (and conformal `solid_mesh` for CHMT) pointing to fresh ASCII polyMesh directories with the same patch identities. Their cell counts are read, hashes retained and actual meshes checked. Do not infer a y+ change from an unchanged imported mesh.

```sh
python "$S/compare_pair.py" /tmp/chmt-native/flatplate_fixed__lowRe \
  /tmp/chmt-native/flatplate_fixed__boundaryLayer_reactingSst --output /tmp/flat-pair.json
```

Pair comparison requires two completed native runs of the same physics/material/end time. Its standalone report retains both frontend hashes and both gas-backend provenance records; old receipts without a backend record explicitly show `UNAVAILABLE`, never the frontend hash as a replacement. Identical stationary meshes compare actual cell fields. Different meshes require actual cell-centre exports (`postProcess -func writeCellCentres -time 0`) and SciPy linear interpolation; extrapolation is rejected. Current moving CHMT CSV lacks physical coordinates: only volume-averaged participant temperatures are compared and spatial profiles are explicitly UNAVAILABLE. This limitation cannot be repaired by inventing coordinates or comparing receded cell labels as identical physical points. All category 3 comparisons are diagnostic, not hard acceptance.

## Independent external references and licensing

See `references/EXTERNAL_REFERENCE_CARDS.md` and checksum-pinned `references/sources.json`. Download scripts store data outside the repository; CC-BY-NC material is never redistributed here.

```sh
python "$S/fetch_references.py" nasa_sst_cf.dat --output /tmp/references/nasa_sst_cf.dat
python "$S/import_reference.py" --source-id nasa_sst_cf.dat \
  --input /tmp/references/nasa_sst_cf.dat --output /tmp/references/nasa_sst_cf.json
# Explicit noncommercial-license acknowledgement is required for AblaNTIS.
python "$S/fetch_references.py" ablantis_exp2_tc.dat \
  --output /tmp/references/ablantis_exp2_tc.dat --allow-noncommercial-data
python "$S/import_reference.py" --source-id ablantis_exp2_tc.dat \
  --input /tmp/references/ablantis_exp2_tc.dat --output /tmp/references/ablantis_exp2_tc.json \
  --acknowledge-noncommercial-license
```

Imports verify SHA256 and preserve headers/units. NASA CFL3D output is a numerical reference; Karman-Schoenherr is a correlation. AblaNTIS has real TC and recession data, but current material law/geometry/mechanism limitations prevent a faithful native experiment case. The temperature angular header says radians while numeric values match the degree-labelled recession grid; import flags this conflict and does not silently convert. Measured TC t=0 is nonuniform. Prescribed surface temperature/recession cannot simultaneously be counted as predicted validation output. No fitting of synthetic material parameters to experimental curves is performed.

## Closed reactive-slab history comparison

The planar closed slab also compares its complete accepted history with an independent constant-rate inventory solution. For surface area A, prescribed mass flux j and declared constant porosity phi, condensed mass loss is A j t and volume loss is A j t / (rho (1-phi)). Current controlled cards have phi=0. This single emitted-component analytic reference is distinct from the general multicomponent endpoint consistency check. With first-order gas conversion S0 -> S1 at rate k, the exact total S0 mass is M0(0) exp(-kt) + A j (1-exp(-kt))/k; S1 follows total mass conservation. The k=0 limit is evaluated without cancellation. These relations do not depend on gas mixing or temperature because the declared synthetic reaction is first-order with zero activation energy.

The report includes every actual/reference history sample and L1/L2/Linf errors normalized by the expected change, rather than by the much larger initial condensed inventory. A frozen no-reaction/no-motion trajectory therefore produces order-one error rather than an apparent perfect conservation match. The 1% objectives are report-only for category 2; missing or nonfinite accepted output remains invalid evidence. Open-flow flat-plate and MSS7 cases do not use this closed-domain reference.


## Wall diagnostic export contracts

The metric reader recognizes gasUGKP `time/uniform/gasBoundaryLayer.csv` and CHMT `chmtOutput/wall-layer-<step>.csv`. CHMT selection uses numeric accepted time and step, preserves the separate stage time, and never reuses an older available snapshot when the latest accepted snapshot reports NOT_AVAILABLE. Species quantities are summarized per species; repeated per-face heat flux and traction are counted once per face. Missing viscosity/distance in the CHMT export means y+ remains UNAVAILABLE. Neither export is converted into an integrated heat rate without actual face areas.

Production probes require the complete declared row shapes and finite outputs. Empty output, missing cases, or NaN cannot pass. The no-device unit test explicitly simulates CUDA unavailability and remains valid on GPU-equipped machines; actual GPU probe execution is a separate requested backend.

## Outstanding cumulative budgets and bounded timing reports

`accepted_cumulative_budget` stays UNAVAILABLE unless a real supported ledger was explicitly requested and read. Neither wall snapshots nor optional last-accepted-microstep owner increments establish final cumulative mass, species, elements and formation-inclusive physical-energy balances. Those require applied face/source integrals, true RK weights, rejected-trial discard and restart-safe ownership, with mechanical work and GCL terms distinguished. CHMT checkpoint ledgers exist but require an independent reader.

For fixed pure mixture-gas cases, set `{"budget_audit": true}` in the case options to request the native opt-in ledger. The runner supplies a fresh absolute `UGKP_GAS_BUDGET_OUTPUT` path. This requires the compatible instrumented backend; missing output fails the requested evidence check. The native producer records finalized positivity-limited physical boundary fluxes at the existing cell divergence, propagates their pending integrals through the same conservative RK save/blend weights, and records actual published chemistry species changes. Only an entire accepted requested interval is published. Internal/periodic interfaces are not external transport. There is no separate chemistry heat source because total rhoE already includes formation energy and chemistry preserves it. Gravity/particles and legacy single-gas are outside this instrumentation scope.

The ledger is a fresh NEW_SEGMENT on every process start, including restart. `budget.py segment1.jsonl segment2.jsonl --expected-end T --output fresh-report.json` marks a chain continuous only after species/thermo/mechanism identity, exact round-tripped binary64 endpoint time and exact serialized initial inventories match. Restart quantization or time differences remain UNVERIFIED with signed jumps, separate segment residuals and preserved raw records; aggregate conservation residuals are unavailable, so restart jumps cannot cancel solver errors. A restarted segment alone never claims history from time zero. The reader preserves serialized reduction precision and independently reconstructs species update and mass/element/physical-energy residuals; chemical elemental drift is reported separately and never subtracted from elemental conservation. Native residuals are diagnostics, not corrective sources or new acceptance tolerances. Host-policy and reader tests are not a completed native GPU conservation experiment. Full diagnostic overhead and actual GPU execution remain separate verification requirements.

`performance.py sample.json --output fresh-report.json` labels an already recorded bounded timing sample; it never runs a solver. The input contains `physical_start_s`, `physical_end_s`, `target_end_s`, `wall_seconds` and optional boolean `concurrent`/`instrumented` evidence. Omitted flags stay unverified. A partial trajectory produces only a diagnostic linear remaining-time estimate, with contention/instrumentation limitations, ETA reliability NOT_ESTABLISHED and no isolated GPU throughput claim. It never authorizes a long run. Assess short post-fix progress before deciding whether original full endpoints are resource-reasonable; a short sample cannot replace their numerical acceptance.

Requested `end_time` must be finite and positive. Canonical or pinned cases reject a different endpoint instead of silently ignoring it; flexible cases verify the generated controlDict and reference manifest agree. Existing exact final-output scheduling and native completion requirements remain unchanged.

## SST candidate remains at its model gate

Independent inventory-moment SST research commit `a50c3f5` (`research/sst-owner-moment-20261011`, `chmt_sst_candidate_20261011`) remains separate from PR12. The complete zero-Q system converges, but the nearby nonnegative small-Q response is infeasible on that branch. Scheduling and budget diagnostics do not resolve the original `SourceStepLimit` failure. Preserve the original failing inputs, thresholds and requested endpoints; the three blowing states, variable density, ALE, rollback, restart and independent-reference comparisons remain pending until a feasible model/interface passes its prerequisite checks.
