# Current paired-energy verification, 2026-10-11

## Current result versus historical evidence

The original loss of small FP32 gas increments is repaired by authoritative
double rhoE/rhoENext and double energy computation. The previously blocking
particle publication mismatch is repaired by persistent bounded J/kg carry and
actual paired publication, including wall/radiation/restart paths. This is a
source-level and HOST-tested result; actual GPU runtime remains NOT_RUN.

Old gas/particle cache-path fixtures deliberately preserve the former mismatch as
causality evidence. They are not the acceptance oracle for the new source model.
The archived patch at7324ae3 and its “do not publish” assessment apply to that
unpaired snapshot. Current acceptance requires the paired implementation and its
lifecycle commits together, plus the root's integrated native build/review.

## Measured source-kernel results

After the secant-Cp stability correction and stable piecewise enthalpy increments:

- Original signed0.02/0.03/0.05 J gas and particle publication regressions pass.
- At dt1e-8, production ordinary gas source changes by−0.0622222423553 J.
  FP32 particle pT initially remains2800 K while carry changes by
  0.000639999825454 J/kg, and the effective temperature responds.
-100,000 actual particle-first production updates exchange about7199.651 J
  absolute particle heat. Actual combined inventory residuals are
  5.8269361e-6 J (CHT32) and4.3945620e-6 J (CHT64), ratios
  8.09e-10 and6.10e-10 of exchanged heat. The independently accumulated
  publication-roundoff bound is0.00536444 J; it is derived step by step from
  gas subtraction ULPs and compensated particle arithmetic, not a tolerance on
  the large background energy.
- The earlier gas-first representation-only prototype retained ±2999.99713898 J
  after100,000 requested±0.03 J increments. Its0.00286102 J physical-reference
  error is double gas accumulation rounding. Its near-zero paired error must not
  be quoted as the production-order result.
- Independent coupled caloric long-double RK4 reference: CHT32 summed temperature
  error falls0.0757382,0.0398403,0.0204627,0.0103769,0.00521829 K for
  2,4,8,16,32 steps over0.02 s. CHT64 has the same first-order trend.
- Stiff unequal mass/rate repeated steps approach represented equilibrium:
  maximum temperature error0.000210569 K in32 (storage ULP0.00048828125 K),
  5.19e-11 K in64. Gas stays above its actual energy/temperature floor.
- The old endpoint-Cp overshoot example (4000 K gas,2800 K receiver, gas
  capacity1 J/K) is frozen in a secant-convexity regression. Stable signed
  increments include sub-ULP intervals exactly at solidus/liquidus boundaries.
- Fused gas recovery is compared bitwise against the real separate gas EOS
  function for E,T,p,U; no old particle moment recovery traversal is needed.

## Final frozen tests and boundaries

The energy-branch verification checkpoint was
`c86f5b1975935838de0089e4238948f0d1198a94`. The source did not change
during the following runs; these counts belong to that checkpoint, not a
claim that a later integrated native build has passed:

- `tests/gas_transport tests/chemistry`: **426 passed, 10 skipped**,215.57 s.
  This includes the actual-order100,000-step, lifecycle, radiation, restart,
  coupled caloric refinement and67 cold-carry/transaction cases.
- Existing cold route/1D energy/CUDA-fixture/launch-default roots:
  **112 passed**,10.33 s. CUDA fixture skips retain their actual gate semantics;
  this result does not imply device execution.
- Managed-mirror, source-contract and operator-ownership regressions:
  **30 passed**,9.84 s.
- An independent extracted-snapshot review: **123** targeted energy, paired,
  cold, recovery, ALE and state tests plus **6** legacy golden tests passed.
- Final EOS extraction targeted run: **15 passed**, including all six legacy
  FP32/FP64 adapter outputs. No further golden hash changes were required.

The earlier changing-tree runs are historical diagnostics, not the final result.
The separate focused cold run before EOS helper extraction passed179 tests;
its cold tests are included again in the final roots above. Schema8 crossprecision
rebasing, active-profile carry, injection/compaction/copy/wall release and material
phase/radiation consumers are exercised by the final gas-transport suite.

The actual1D wrapper test exposed an independent-lane race: material mass and
geometry must be frozen in every lane before lane-zero gas publication. Commit
`a18d847` adds both required warp barriers. It also replaces a CUDA-invalid device
`numeric_limits::epsilon()` call with exact binary32/64 constants. The two first
native compile failures are retained as RED evidence; they are not final builds.

The cold shared-carry review found both negative- and positive-heat counterexamples
where nearest-float mean rounding made the smallest effective node negative.
`0809a7f` publishes nodes downward, preserves the actual mean with bounded carry,
and includes the matching double-arithmetic cooling margin before credit is
shared. Both original probes now preserve positive effective nodes. Registered
mixed ordinary/1D/2D transactions verify actual negative credit, actual positive
publication, wall-separated budgets and gas floors.200,000 logarithmic8/64-node
proposals per precision verify no accepted candidate is lost at publication.

`c86f5b1` shares the existing caloric floor and kinetic arithmetic with paired
acceptance. With rho=rhoMin=1,Rgas=1e-4,gamma=1.4,Tmin=1 and gasE=1.1,
the prior heat floor0.00025 allowed recovery to add energy. The real EOS floor1
is now reserved before publication; actual recovery adds exactly0 in both
precisions. Normal thermal floors, nonunit density/nonzero momentum and small
gamma−1 cases are covered. Existing homogeneous fmax fallback semantics remain.

The actual complete strictAdvance CUDA fixture covers configured L0/L1 and
Euler/RK2/RK3, no heat/microheat/finite heat. Its inventory includes carry. It
includes real allocation, cube geometry, zero face flux checks, collision pool,
tracking, compaction and finalization. No stages are stubbed. Native compilation
is delegated to the root's serial queue; actual GPU execution and atomic-order
measurements require a GPU and remain NOT_RUN. All HOST atomic adapters are
sequential and cannot certify real GPU reduction ordering.

## Later traversal fusion

The integrated source was then optimized at `cd5074b` without changing accepted
heat arithmetic. Existing pool clear and post-sampling correction loops now
prepare heat; selected stuck particles contribute only after their finalizer.
The real sampling/correction numerical fixture passes24 cases across32/64,
including invalid pools, cold detachment, signed heat, reversed reduction order
and compiled heat-disabled paths. The focused combined source-order, complete
advance fixture and production heat run passes42 tests with4 retained CUDA
skips. The independent reviewer passes109 tests with4 CUDA skips. Existing
cold routes/energy/launch fixtures pass113 tests.

`dc25e4d` additionally removes one restart-only host scan by checking carry
inside the existing upload particle validation loop. The extracted real-loop
regression was RED in both precisions before fusion. The final restart/lifecycle/
authoritative-energy group passes11 tests, including nonfinite carry at every
position and preservation of the existing invalid-mass error before particle
array uploads. Schema8 field-major codec validation and post-read FP32 rebasing
remain separate low-frequency operations: combining those would change the
codec API or frontend validation/partial-mutation ordering, beyond adjacent-loop
fusion. No hot-path pass depends on them.

The final frozen `dc25e4d` gas-transport/chemistry run passes **560 tests with
14 skips**,289.96 s. Production source did not change during that run. The317
managed mirrors and all four solver/precision field-lifecycle closure probes pass. Native compiler resources and exact build provenance are in
native-compiler-costs.md. The root's integrated `43da9c8` complete strictAdvance
fixture compiles in both physical32 and64:2 tests pass,14 unrelated cases are
deselected,113.32 s. This is compile-only and actual GPU execution remains
NOT_RUN.

## Integration and cost

See paired-carry-design.md for all allocations, exact launches, cold2D conservative
reservation and finite-step limitations, and pass-fusion.md for the later
post-collision fusion. The final fused schedule has zero net added traversals or
launches after replacing the old Eulerian heat/recovery pair. Candidate reduction
and scratch clear now run inside existing correction/pool-clear traversals; their
arithmetic, atomic operations and data traffic remain real costs. The energy
branch originally used `ab900f54` as
its base. The integrated PR12 candidate now includes the newer gas budget,
resource-bounded geometry and numerical SST wall-function changes together with
paired thermal carry. Final integrated native compilation must be recorded
separately; actual GPU execution remains NOT_RUN. Historical EOS/RK identity
repairs are independent behavior fixes, so the full patch is not advertised as
globally bitwise unchanged.
