# Bounded transport repairs, 2026-10-11

Scope: repair the previously demonstrated surface-identity, rounded-publication,
zero-inventory closure, microstep-tail and inactive-diffusion-channel defects on
PR7 before integration into PR12/13. No README changes, geometry constraint redesign,
SST model redesign, relaxed tolerances or full-application FP32 acceptance.

## Design and test sequence

1. Surface recession: retain the volume formula and build a validated candidate-local
   persistent-ID index (the initial LP03 port used a packet-by-surface search).
   Resolve each GasSolid packet by persistent surface ID exactly once, reject missing
   or ambiguous matches and inconsistent/out-of-range solid cells before indexing.
   Use each condensed constituent's actual density and porosity recovered from the
   actual donor inventory and volume. Multiple distinct time packets for the same
   valid surface remain legal. Test the actual production loop in a compiled host
   extraction and run the native OpenFOAM material regression when available.
2. Chemical publication: retain the original input closure gate and reactor model.
   Reconstruct a populated dependent partial density only for the bounded conversion
   roundoff identified by LP04. Re-audit actual published masses, elements and energy
   before writes; zero interval is bit preserving. Test nonbinary stage volumes,
   publication rejection/atomicity and the existing chemistry adapter/reactor suite.
3. Transport publication: add the LP21 bounded local Ns closure in the existing
   cell update, using old/new volume scaling for ALE. Test repeated conservative
   updates, varying stage volumes, malformed incoming states and flux defects.
   No closed-reactor elemental invariant is imposed on an open transport cell.
4. Species closure: preserve the populated-last arithmetic, redirect only an absent
   dependent component, retain legitimate one-sided inflow. Sum diffusion enthalpy
   from the final corrected fluxes. The convective rule is narrower than old LP08,
   which always chose the largest component. Dedicated FP32/64 host operator tests.
5. Microsteps: port LP16's two actual integrations for a would-be sub-minimum tail.
   Keep cap/max/min constraints, representable positive time advance and retry
   halving. Test feasible/infeasible partitions and shrinking retry caps; do not
   infer that unknown future stability limits can always permit the endpoint.
6. CFL: retain the existing geometry gate and face loop, but activate bounds only
   for channels that the actual face operator evaluates. Adiabatic slip has none;
   fixed-temperature slip has thermal conduction but no viscosity/species diffusion;
   species diffusion exists only on internal/periodic or composition-fixed ordinary
   boundaries. Keep conservatism on the active channels and test each in isolation,
   including ALE geometry rejection. This extends CONT03 and requires new red/green
   channel tests; old full-case timing extrapolations are not post-fix measurements.

Each change requires a failing behavioral test before production edits, a focused
passing rerun, and affected regression evidence. CPU execution is not native CUDA
proof. Full native GPU application validation remains a separate gate.

## Review refinements before publication

Transport publication now checks every final face's actual species sum against its
actual total mass flux before cell accumulation; opposite face defects cannot hide
in a net-cell cancellation. The checks are fused into the existing face/species
reads, retaining each species' original face order and reusing the existing Ns
publication array as accumulation scratch. No additional face/domain traversal or
Ns-sized scratch array is introduced. The local operation-count bound excludes the
incoming closure defect, and the final stored representation must pass the original
density closure gate. Negative incoming partial densities are rejected even if an
inflow would make the proposed state positive. A tiny positive dependent component
whose required replacement is negative falls back to the largest populated carrier.
Tests independently read actual published ALE mass, two element budgets and physical
energy against the finalized face transport, with precision-specific rounding bounds;
these are not a full-application cumulative ledger or FP32 increment-accuracy claim.

The ordinary-face CFL change requires an integration guard in PR12: a boundary-layer
face slot bypasses the ordinary physical operator and can publish heat/species flux
without ordinary boundary flags. Its existing conservative bound must be preserved
until that separately evaluated wall exchange has a proven channel-specific bound.

Legacy-state golden refresh is permitted only after restoring the frozen CFL kernel
alone reproduces every old hash, and per-array hashing proves that only
`gasDiffusionNumber` changed. Numerical channel tests remain independent of hashes.

The final mapping design builds one transient persistent-ID index per material
candidate, including inactive/unreferenced surfaces, and rejects duplicate IDs at
construction. Its lifetime is one `advanceCandidate`; no cross-candidate cache is
introduced. Base and endpoint topology/addressing must agree, and each packet
lookup checks the current indexed address. This replaces packet-by-surface searches
with expected constant-time lookups; the single O(surface-count) construction and
bounded host map allocation are explicit costs and are not a new gas-cell sweep.

Acceptance and repair are distinct: valid input and output still use the unchanged
64-epsilon density gate. The input discrepancy is transported in integrated mass
units, not added to each step's repair allowance. Only newly introduced arithmetic
drift can be corrected; a zero-flux accepted state must remain bitwise unchanged.

The publication scratch is not larger, but its longer live range and extra bounded
scalar arithmetic can affect register pressure. CUDA resource counts and native GPU
performance have not been measured in this host-only verification environment.

## Native build and compiler resource audit

The standard gasUGKP CUDA backend plus OpenFOAM 10 frontend compiled and linked
with CUDA 13.1.80, GCC 14.2, `sm_89`, Ns=2, binary64 and FMA enabled. The first
native build rejected a host-only `numeric_limits::epsilon()` call in the new
cell update; exact binary constants corrected this without changing build flags.

A same-toolchain `cuobjdump --dump-resource-usage` comparison against `ed94bc1`
shows nonzero compiler costs. The cell flux update changes from 46 to 48 registers
and from 0 to 16 bytes of stack. Diffusion estimation changes from 128 to 129
registers, with its 144-byte stack unchanged; shortening the thermal-channel flag
lifetime reduced an initial 130-register result to 129. Chemistry remains at 255
registers, with stack changing from 1744 to 1776 bytes. Two internal-face kernel
instantiations change from 218/255 to 220/255 registers and from 352/336 to 400/368
stack bytes. Primitive gradients remain at 164 registers and 48 stack bytes.
These are compiler resource counts for this target, not timing or a universal
precision/species-count result. Register allocation thresholds can affect resident
blocks; actual GPU phase and solver throughput must be measured. No GPU runtime
is available in this build environment. The reduced inactive-channel timestep
restriction and fewer host searches cannot be substituted for that measurement.
