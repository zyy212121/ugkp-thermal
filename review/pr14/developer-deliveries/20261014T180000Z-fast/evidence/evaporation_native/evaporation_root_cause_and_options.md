# FAST4 evaporation: confirmed mechanism and native design options

Production source: `9d5be6eff69dbced2fc959d422d3ba419a1bc39e`. Initial diagnosis and candidate experiments were isolated in scratch. After explicit root review/authorization, exactly CpuFilmDriver.C and its test_cpu_film.cpp regression were applied to the shared tree. No commit or GPU run was performed here.

## Confirmed evidence

- The fresh formal case timed out at 900 s with committed time 0.00025 of requested 0.001. The D302 diagnostic observer accepted 4,276 local material candidates but paid only 4.861239503690122e-13 s of local time.
- `verify_evaporation_packet_sequence.py` reproduces every accepted dt exactly: downward-rounded remaining retry-12 span divided by 2^30. All 4,276 comparisons have zero ulp error. The first retry-12 actual gas interval is 1.2207031249998656e-7; first material dt is 1.136868377216035e-16.
- All 17,104 face packet mass overlaps equal the original gas packet mass multiplied by the overlap fraction, with zero ulp error. This is not evidence of an inflated evaporation rate. The gas-forcing mass-depletion scale is about 3.70118 s. A tiny accepted subtraction is only 0.17706 ulp of the stored film mass; all observed film masses/enthalpies remain unchanged.
- The prior available checkpoint bytes exactly match the fresh formal and observer checkpoint hashes. Reuse here establishes identical checkpoint content, not a fresh execution. The native probe decodes the full serialized ModelConfig and HostState using production Checkpoint.C, and links actual production CpuFilmDriver.C.
- Native probe with checkpoint per-face pressure reproduces `CPU film undeclared open-edge inflow at 0` at the full retry-12 interval and at divisions by 2, 2^10 and 2^30. The checkpoint bulk-gas EOS pressures are bit-identical and give the same result. Existing terminal completion succeeds at all these dt. A uniform-pressure diagnostic control passes all ordinary operators.
- Per-face pressures differ by only two ulps (about 5.82e-11 Pa). Production transport recomputes the stored face-0 gradient and tangential mean velocity exactly. Edge 0 has owner 0, neighbour -1, and length rate -1.939993272230143e-19. The sign qualification does not depend on dt when the geometry is frozen.

Probe scope is intentionally narrower than full-owner replay: endpoint pressure and tangential traction are reconstructed controls, geometry is frozen, and some unlogged forcing fields are reconstructed. D302 does not record all stage gas traces/gradients/matching/geometry required for a bitwise original history replay. Thus the exact original first rejected branch remains a GPU/complete-history observation gap, while a matching native, dt-independent failure mechanism is reproduced.

## Exact source chain

1. `gpu/Backend.cu:429-447` computes current classical gas Courant and selects dt before executing the actual gas trial. CPU repair need not alter that owner or replay gas at a larger interval.
2. `io/MultirateEvolution.C:117-129` invokes terminal CPU material completion after numerical failure at the exhausted macro retry budget, preserving the actual attempted gas interval and history.
3. `materials/CpuMaterialDriver.C:320-324` repeats `max(maxRetries,maxCouplingIterations)=30` local attempts before setting terminalLast. The four halving categories are reaction (547), packet application (554), material recovery (572), and film candidate (578). The original diagnostic is cleared at every retry.
4. Physical selections happen separately: material transport at 337 and 537, film donor/thermal/transport estimator at 438-444; source-history alignment at 387-415 and geometry/rate consistency refinement at 508-517 are separate mechanisms. Geometry failures themselves break or return; they do not directly execute those four half-step statements.
5. `film/CpuFilmDriver.C:253-260` constructs the signed edge length rate, rejects any negative open-edge value in the ordinary path, then already chooses the owner reconstruction when neighbour < 0. Existing terminal evaluate runs that signed owner-extrapolated expression.
6. The film candidate re-evaluates ordinary transport at 353 even after the material owner earlier used terminal transport completion. The full film completion is delayed until CpuMaterialDriver.C:577 terminalLast. This is sufficient to generate the observed repeated 30 halvings for the reproduced sign qualification.
7. Film donor/stability formulas are at `CpuFilmDriver.C:459-484`; actual mass/species inventory checks are at 359-363. They must not be conflated with the open-edge sign qualification.

## Safety and boundary semantics

`SurfaceMesh` (core/HostState.H:19-26) has no film edge boundary-type or exterior-state field. `io/MeshIO.H:256-287` uses neighbour -1 for a missing neighbour and does not declare a zero-gradient inflow condition. Importantly, `devtools/multirate/test_cpu_film.cpp:138` explicitly expects undeclared edge inflow to be refused. Therefore deleting the sign qualification globally expands the ordinary path from outflow-only to signed owner-extrapolated open-edge flux. The arithmetic already exists in terminal completion, but this must be stated as a behavior change, not described as an already-declared normal boundary model.

The `numericalFailure` bool is not a sufficient proof that completion can be safely moved early everywhere. MaterialTransport.C:108 and CpuFilmDriver.C:360/363 label actual donor exhaustion numerical as well. The broad four-helper immediate-completion scratch patch is an unapproved design specimen, not a safe validated fix.

## Options considered and selected scope

A. Selected for scratch red/green verification under root-confirmed user authorization: remove only the negative-open-edge sign qualification, retain finite/input handling and all candidate donor/EOS/accounting checks, and continue using the already-written owner extrapolation. No new thresholds, physics parameter changes, kernels or traversals. Ordinary previously successful cases execute identical expressions in identical order. The open-edge behavior expansion above is explicit, authorized scope; native donor/nonfinite/missing-input controls remain required.

B. Alternative not selected (extra architecture unnecessary for the authorized narrow boundary change): separate terminal transport evaluation from ordinary film-candidate checks (e.g. `<TerminalCandidate=false, TerminalEvaluation=true>`). Call it only after an ordinary terminal-owner film failure. Keep actual donor checks and endpoint EOS ordinary, keep the existing final completion unchanged. This avoids globally changing the normal open-edge contract and does not promote all numerical failures to unrestricted completion. It adds no success-path traversal; a recovered failure uses one additional existing film evaluation, instead of 30 full material restarts. It requires extra code/template surface compared with A.

C. Broad four-helper fallback acceleration, rejected as unnecessarily broad: the scratch `immediate-completion-candidate.patch` uses current dt, so no tiny-stage geometry is reused and all previous stages belong to that same dt. It preserves nonterminal success arithmetic and does not broaden terminalLast physical-limit exemptions. However, because donor failures share the numerical flag, this option is not recommended without separating the physical cases.

Any design that restores a larger dt after halving must rebuild from the accepted local state: interval slice/cursor, transport, gas/surface/phase forcing, physical limit selection, target sweeps and moved geometry, consistent film rates, material sources/packets/recovery, film candidate, endpoint program and ledger. Reusing the tiny candidate and only changing its clock is invalid.

## Traversal / precision / gas-Co impact

The scratch expression option adds zero traversals, zero kernels and no FP32. Candidate template separation adds no work to a normally successful path and only existing CPU film evaluation on a recovered terminal failure; it avoids the repeated whole-stage material re-execution. Neither changes classical gas Co selection or gas donor reservation. The old terminalLast exemptions from material/film dtLimit remain a distinct concern; no proposal should globally turn terminalLast true or use it to bypass all physical selections.

## Scratch native red/green result

The selected finite-only guard candidate is `CpuFilmDriver_signed_open_edge.C`; the frozen source and regression have now been applied to the shared tree after root review. `shared_application_receipt.json` records original and applied hashes; `fast4_signed_open_edge.patch` is the exact two-file patch. `signed_open_edge_execution_receipt.json` records original and candidate compilation/execution, sources and peak memory. Actual ordinary evaluate/estimate/advance now succeed at the full retry-12 interval for both checkpoint-derived pressure controls, and the estimator returns the requested interval. All four dt choices pass. Mass and species donor exhaustion, missing coupled trace/storage, and nonfinite length-rate controls retain the same failures.

Independent comparison (`independent_native_output_comparison.json`) finds all 17 shared successful output snapshots identical field-by-field, including binary64 bit patterns. The eight newly successful ordinary checkpoint candidates exactly equal the original terminal completion output at the SAME dt. FilmQ, FilmAux, Budget, edgeFlux, report energy/residual/kinetic/roundoff fields and every rateBudget field are included. This demonstrates reuse of existing terminal arithmetic with ordinary physical candidate checks retained; it is not a full GPU execution claim.

## Remaining validation

- Durable signed-inflow and independent two-ulp-pressure CPU regression red/green checks PASS under the original strict C++14/O2 flags, with unchanged existing numerical tolerances. The baseline first fails at the two-ulp transport case with the actual native open-edge error. The candidate passes the entire film helper. The unchanged baseline helper also passes against original source. See cpu_film_regression_receipt.json.
- The original check_cpu_film.sh three-component script PASSED on the applied shared-tree files: film, curved film, and phase lifecycle. Exit 0 in 46.36 s; peak child RSS 318688 KiB. All captured source input hashes were unchanged before/after. See check_cpu_film_all_components.log and check_cpu_film_all_components_receipt.json.
- To identify the exact original failed branch, use the scratch read-only `retry-observer.patch` in a real GPU owner run or capture the complete original interval history. Observer changes no solver arithmetic and limits only diagnostics.
- A production change requires native full-owner controls and fresh same-input GPU completion/progress evidence. This investigation does not close the original .001 evaporation completion gap.
