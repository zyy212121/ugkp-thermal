# CPU film report-removal test migration

Own files only: applications/CHMT/devtools/multirate/test_cpu_film.cpp, test_cpu_film_curved.cpp, test_cpu_phase_lifecycle.cpp. No production, README, material-native fixture, Ledger/Coordinator test or script changes. Existing signed-open-edge evidence under pr14_fast4_evap_analysis is untouched.

All candidate.report references removed because CpuFilmCandidate now contains only actual film and filmAux. No zero budget assertions substituted.

Physical checks retained/rebuilt:
- Test-local U sum is directly H - p*M/rho from actual old/new inventories (not a runtime report or helper total).
- Pressure storage is independently p_new*M_new/rho - p_old*M_old/rho; verify H shift, unchanged T and unchanged U.
- Surface forcing checks actual U change = physical source + radiation + prescribed kinetic channel, and actual mass unchanged. No removed diagnostic kinetic defect is treated as physical heating.
- Signed open-edge outflow/inflow retains actual mass, species, enthalpy and elemental inventory tests. Actual transport RHS is inspected via existing evaluateCpuFilmTransport, then compared with independent owner extrapolation, pressure work and U closure.
- Two-ULP pressure perturbation at full dt and 30 halvings retains strict signed negative boundary flux from actual transport RHS; compares with rho*thickness*profile length-rate, accepted dt equality, actual mass/species deltas and independently reconstructed physical-energy closure.
- Radiation cooling compares actual U change with the independently written Stefan-Boltzmann formula.
- Body/support/top work checks actual endpoint mean/top velocities against an independently obtained physical profile and U against total mechanical work, without reading removed reporting totals.
- Curved transport checks actual RHS mass/species/enthalpy flux against analytic shared-hinge geometry and kinetic flux; antisymmetric cell RHS and actual inventory conservation; external shear work versus actual U; angular convergence unchanged.
- Phase lifecycle event is inferred from old/new actual zero/nonzero mass. Test-only phase storage/residual is reconstructed from actual states, actual transport RHS and forcing. Existing gasMass/gasEnergy/phaseMass helper fields and main entry remain available for support worker physical parity.

Existing FilmRateBudget values read in tests are actual transport-RHS outputs, not recreated production reports. Independent analytic/state comparisons prevent merely checking a new zero placeholder. No GPU/timing claims; check_cpu_film.sh builds and runs three host integration binaries.

## Final API removal update (17:26 UTC; supersedes intermediate RHS-budget observations above)

Production also removed FilmRateBudget and the unused Evaluation.budget/edgeFlux vectors. All owned tests now use only actual transport `rates`, actual profiles and inventories; no budgets argument/type remains. Edge energy in no-source fixtures comes from the signed receiving/owner RHS and is checked against independent owner enthalpy flux; two-ULP closure uses the sum of actual energy RHS. Kinetic-flux-only output checks are removed because that path was not used in the state update; actual endpoint kinetic profile checks and physical thermal-energy closure remain.

Additional ownership authorized: tests/fixtures/film_incidence/probe.cpp, tests/test_film_incidence.py, and only test_original_film_bytes_and_costs in tests/test_terminal_sources.py. Historical signature adaptation exists only in test-generated C++ and executes the old physical RHS without changing it. Physical stdout must remain exactly equal. Allocation differences are checked explicitly, not ignored. Full immutable old baselines 6dd13b and 8ed8a are unavailable; original tests are BLOCKED at Git reads (logs incidence.log and terminal_cost_before.log), not passing. A separate within-this-task saved-source comparison is prepared to measure the actual deletion cost without pretending to replace missing historic baselines.

The initial three host components passed with report removed but before the final budget-output API deletion (host_components_final.log). That intermediate PASS does not certify the final API. Final runs await the parent's CPU compile window.

## Final verified results (17:30–17:31 UTC)

- Final-API `check_cpu_film.sh`: all three host executables PASS, including signed-open-edge/two-ULP, curved transport and phase lifecycle. Log final_api_host_components.log.
- Actual historical-checkpoint native CPU replay: 17 original successful cases, 8 newly admitted ordinary cases equal original terminal same-dt cases, and all 25 prior signed-edge successful cases have exactly equal binary64 film/filmAux fields. No deleted reporting fields/checksum are compared. Results native_replay/physical_comparison.json; executable input/source hashes and commands recorded. This is frozen geometry/recorded forcing, not a full GPU owner replay. Original evidence is untouched.
- Within-this-task actual saved-before-source versus final-source comparison: all physical stdout identical for 32 size/order/reconstruction/sign combinations plus limited-retry/endpoint/isolated witnesses. Every traversal counter and other stderr line remains identical; only measured COUNT and ADVCOUNT allocation data changes.
- Actual allocation measurements for 24,48,96,192 faces (edges=2*faces): evaluate allocations15→13, bytes saved3072/6144/12288/24576; advance40→36, bytes saved6144/12288/24576/49152. Verified exact formulas: evaluate64*faces+32*edges; advance twice that. Saved32 bytes/edge is sizeof(FilmQ) at this Ns2 test configuration; saved64 bytes/face is former FilmRateBudget. Evidence cost_physical_comparison.json, final_api_costs.log, cost_before/ and cost_after/ stdout/stderr/build logs.
- Missing immutable 6dd13b and8ed8a historical regressions remain BLOCKED. The new saved-source comparison is explicitly not a substitute or pass for them.
- All11 recorded source hashes remain identical from run start to completion (final_source_hashes.txt). `git diff --check` passes for owned files. No real-GPU arithmetic or speedup percentage claimed.
