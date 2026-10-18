# OpenFOAM diagnostics and single-pass advancement

Status: full native build passed at 9ed4ff10; the C++14-only client correction edc536ef was rebuilt and produced identical binaries. See 工作报告.md for exact evidence and 遗留与例外.md for acknowledged gaps. Post-publication audit is pending. This report does not certify absence of every retry path.

## Requested semantics

- Actual OpenFOAM host FatalError/Warning handles solver diagnostics. CUDA records a structured first fatal and bounded warning summaries in per-instance POD storage.
- Physical/runtime failures stop dependent local arithmetic and terminate the host workflow at an existing completion/readback boundary. No physical-step retry, rollback-and-replay or alternate terminal solution is used to recover.
- Finite conservation/accuracy discrepancies warn while preserving the computed result. Finite mass fractions are explicitly bounded to [0,1]; nonfinite inputs are not converted to a finite fraction. No general composition renormalization is introduced.
- Preserve original direct Riemann flux fallback, positivity/floor definitions, mathematical Newton/root/linear-solver iterations, physical time loops and RK/Strang stage sequencing. Particle traversal/inactivation behavior is excluded from this change.
- No backup algorithm implementation is introduced. Git history retains previous source.

## Implemented groups

| Group | Changed behavior | Diagnostics |
|---|---|---|
| Shared gas requested interval | One requested advance; no dt-halving/doubling retry, snapshot rollback or terminal continuation | Existing completion boundary observes structured device failures before synthesizing missing-storage status |
| Gas chemistry | One SDIRK integration, ordinary mathematical stages retained; no full/two-half estimator or damped candidate loop | Physical failures are fatal; finite convergence/conservation residuals warn |
| Mixture reconstruction/thermo | One limited reconstructed state; no alpha acceptance; no failed evaluator or saved-temperature terminal result | Finite Y clipping warns; nonfinite state and invalid calorics report failure |
| Wall layer | Full Newton correction; one Jacobian perturbation per column; no alternate seed/terminal profile or failed residual replay | Physical evaluation failure records diagnostic; finite nonconvergence warns |
| CHMT reaction/material | One full Euler reaction at preselected physical step; no step-doubling estimator/local failed-step retry | CPU per-operation context dispatches error/warning |
| CHMT coupling | Removed predictor replay, shortened retry windows, rollback consumer and conduction-continuation alternative | Coupling numerical residuals warn; unavailable physical result fails |
| CHMT surface/film | Removed surface composition alpha loop, failed-root alternate feasible-domain continuation, film reconstruction alpha loop, complete/terminal implementations | Explicit fraction bounds warn; finite composition/root residuals warn; invalid evaluations fail |
| CHMT moving geometry | Full correction per mathematical iteration; no backtracking or permissive geometry completion | Finite GCL/sweep/accuracy residuals warn; invalid topology/nonfinite/nonpositive geometry fails |
| Exchange packets/inventories | Finite mass/energy decomposition mismatch no longer rejects | Actual discrepancy recorded; exact computed extensive quantities retained |

## Specific surface/caloric changes

`InterfaceMath.H` retains mathematical bracket discovery and bisection. An invalid sampled state or midpoint now ends the solve immediately; it cannot be followed by a search for another feasible domain. Exhausted finite bisection/composition iterations retain their current evaluated result and warn. The interface composition solve uses one full Newton correction and explicit Y bounding. Its normalization equation is part of the existing coupled system, not post-processing renormalization.

`Thermodynamics.H` removes separate `completePredictorGas` and `completeConservativeGas` implementations and their import/output/wall-sampling recovery callers. Gas primitive/conservative conversion clips finite composition fractions and warns on finite sum mismatch. Caloric recovery preserves the existing mathematical root safeguard and roundoff endpoint convention; no failed-then-terminal re-evaluation is performed. Film recovery treats finite mass-sum discrepancy as a warning.

## Communication and lifetime

See `../diagnostics-transport-boundaries.md`. Host/device transfers occur at existing completed readback/synchronization opportunities. Diagnostic bytes and a bounded companion transfer may be added there. Ordered metadata setup may add a tiny device kernel but no synchronization or new IPC operation. Source descriptors accompany only reported sites. Metadata and first-fatal lifetime must be validated against graph replay, teardown and multiple solver instances.

## Validation boundaries

Actual CUDA/OpenFOAM compilation is required before publication. No GPU is available in the execution environment; CUDA runtime numerical validation, race/synchronization sanitizer checks and production performance are not claimed. CPU probes do not establish GPU correctness.

Old tests which explicitly assert successful NaN/negative-temperature publication, full/half retry, terminal fallback or conservation rejection encode superseded behavior. They must be identified individually; failures are not automatically dismissed as obsolete. Larger valid-input cases that cease to solve after removing retries must be reported separately from fixture/API updates.

## Remaining mechanisms and unintegrated paths

Pending final source audit. All retained or unresolved paths must list file/function, actual behavior, whether reachable in production, diagnostic coverage and reason for retention. Compatibility wire/checkpoint slots do not by themselves imply an executable retry algorithm; their runtime consumers must be checked.

## Post-publication audit

Pending. Audit must identify the exact PR head and compare the entire tracked library, including configurations and inactive compiled variants, rather than only the changed files. Allowed direct flux fallback, mathematical iterations, particle exclusion and infrastructure I/O retries must be separated from prohibited numerical candidate/retry behavior. New findings are reported and appended; no absolute absence promise is inferred from a keyword scan.
