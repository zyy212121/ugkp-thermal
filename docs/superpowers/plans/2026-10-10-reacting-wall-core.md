# Reacting boundary-layer implementation and verification checklist

Goal: a transactional shared host/device reacting wall closure, with the outer gas as the sole conserved-state owner.

Technical contract: [reacting-wall design](../specs/2026-10-10-reacting-wall-design.md).

## Implemented core
- [x] Explicit model family, scalar types, borrowed thermodynamics/mechanism/geometry views and reusable bounded workspace.
- [x] Frozen constant-property heat/shear limits and atomic failure publication.
- [x] Conservative velocity, total-energy and differential-diffusion species balances with finite-rate chemistry and formation-inclusive enthalpy.
- [x] Positive Picard initial guess and damped block-tridiagonal Newton solve; no frozen-chemistry fallback.
- [x] Feasible simplex-tangent derivatives for exact-zero and locally exhausted species.
- [x] Table-valid temperature constraints and explicit missing-coarse-chemistry diagnostic availability.
- [x] SST conservative shared-face omega flux, smooth-wall asymptotic fitting, physical-omega cross diffusion and Wilcox positive-blowing boundary condition.
- [x] Continuous near-wall k reconstruction, finite owner omega, physical wall k flux and source reconstruction consistent with BVP nodal sources.
- [x] Actual nonlinear failure residual reporting and explicit native-precision SST requirement.

- [x] Wall-relative velocity, compensated temperature and formation-preserving common enthalpy reference; cached NASA branch offsets and actual-step feasible Jacobian differences.

## Executable core evidence
- [x] Frozen zero/finite-blowing limits, mechanical-heating independent Green integral and invalid-input atomicity.
- [x] Reacting diffusion against a closed-form solution with frozen control.
- [x] Exothermic differential diffusion against an independent SciPy boundary-value solution and nodal refinement.
- [x] Exact-zero ten-species creation, exhausted-dependent simplex corner and strong reaction.
- [x] Local-Cp conductivity law and thermodynamic range checks.
- [x] Independent long-double integration of all ten generated H2O2 NASA species, tiny and cross-branch increments, allowed enthalpy jumps, differential-diffusion formation energy and two-component mechanical-frame restoration.
- [x] Original SST k=.5/omega=200 counterexample, zero/tiny/finite blowing, physical owner volume and first moment.
- [x] Near-wall bridge against an independent advection ODE; physical-omega cross-diffusion derivative regression.
- [x] Owner source agrees with the converged BVP node source at exact-node quadrature points.
- [x] Separate R=GpuReal FP32 and FP64 executables. Strict-default FP32 failures remain explicit; separately configured tolerances are reported with achieved residuals and FP64 output comparisons.

- [x] FP32/FP64 20-case SST precision comparison, Galilean shift, strong Ns10/simplex corner and simultaneous reaction/SST/blowing smoke.

## Remaining acceptance work
- [x] Profile-aware actual-polyhedron quadrature and a full-SST cube comparison against independent per-profile-interval Gauss32; finite-grid source/flux discrepancies remain documented.
- [ ] Final fixed-revision independent residual/Jacobian/flux/source audit and application regression results.
- [ ] Native CUDA runtime evidence on a GPU; compilation alone is insufficient.
- [ ] Wall-resolved same-model space/time convergence and the physical validation cases listed in the technical contract.
- [ ] Broader reacting/SST parameter-range validation, especially stronger blowing and source terms with nearly cancelling production and destruction.

A passing nonlinear residual or a unit-test suite does not close the remaining accuracy and physical-validation items. Preserve those limitations in the review record.
