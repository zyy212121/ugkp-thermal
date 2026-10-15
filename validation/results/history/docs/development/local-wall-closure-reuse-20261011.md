# Identical wall-closure reuse within one surface-temperature evaluation

The composition fixed-point loop can request the same wall-layer closure repeatedly while relaxing its separate composition iterate. The cache is automatic local storage inside `evaluateSurfaceTemperature`; it does not survive a call, stage, trial rejection, geometry update, restart, or temperature evaluation.

The exact key covers the quantities that can change inside this loop: all species mass fluxes, wall-normal speed, and all components of wall velocity. Bulk/gradient, candidate top temperature, distance/normal, flux tolerance, physics configuration, matching state and wall context are constant during this call. Only successful closure output is reused. Different trial fluxes or velocities force a new solve. No approximate equality or residual tolerance is introduced.

## Verification

The instrumented actual-header test initially observed 31 closure calls for one accepted fixed-flux evaluation. The same evaluation now calls the closure once. Six independent evaluations vary temperature, matching geometry, conductivity, species flux and velocity. Every output printed at 17 significant digits matches an uncached reference; every independent call performs its own closure solve. The reference disables only this local reuse, retaining the same production equations.

Local reuse and existing wall regressions: 15 tests passed. Six additional frozen-mixture, SST-disabled BVP evaluations also match uncached physical outputs and active workspace state/residual; their maximum residual is 1.38e-10, below the unchanged 1.01e-8 threshold. A production surface reaction changes species fluxes during the composition loop: both ConstantTransport and BVP paths reevaluate correctly. Disabling the cache-key comparisons makes those two regression tests fail. These BVP tests do not establish SST or chemistry model acceptance. This is a host correctness and invocation-count result, not a measured GPU speedup. Full application integration and GPU runtime remain separate verification requirements.

## Cost

Cache state: one boolean, `Ns+1` real values, and one three-component vector, plus the loop-local `sameClosure` boolean. Compiler register allocation is a separate measurement. Each composition iteration compares at most `Ns` species fluxes and four scalar motion values. No heap allocation, persistent state, GPU kernel, synchronization, host-device transfer or extra domain traversal is added.
