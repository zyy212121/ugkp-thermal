# Authoritative gas energy and paired thermal publication

## Approved scope and representation

This local repair retains small gas source increments and pairs actual particle
thermal publication. It prioritizes stability, conservation and efficiency.
It does not promise exact FP32 momentum conservation or recover energy already
lost in old checkpoints. The earlier gas-first carry prototype and blocked
migration evidence are historical; production uses particle publication first,
then the opposite actual gas subtraction.

CHT `rhoE` and `rhoENext` are double authorities independent of transport Real.
Mass, momentum, flux input arrays and particle temperature/profile arrays retain
their existing types. The shared typed gas state carries a separate Energy
parameter. Source, divergence, RK, recovery and thermochemistry read that type;
there is no authoritative float shadow. CHMT and gasUGKP already own double E.

Each CHT particle stores one double residual in J/kg, including its compaction
copy. Ordinary thermal authority is H(stored pT)+residual. Active cold authority
is mean(stored float node H)+residual. Effective temperature and phase evaluation
consume that authority. Publication reconstructs the representable temperature
or profile and retains only its bounded mean roundoff; exactly zero heat is an
identity operation. Allocation, initialization, injection, copy/swap, wall-state
transitions, radiation, wall heat and restart participate in this lifecycle.

Existing ordinary and cold caloric laws retain their established state-dependent
selection. Their raw H difference includes a reference/model difference and is
not automatically a physical heat leak. This patch does not introduce a global
phase-law change or hide that difference in a huge carry. Actual transition tests
compare identical zero-carry baselines: pT, profile and Cp keep the legacy
behavior while the newly represented signed J/kg increment survives unchanged.
Schema7 reads initialize carry to zero. Schema8 preserves it; ordinary pT
precision conversion rebases the carry, whereas a cold profile's derived pT
never replaces node authority.

## One fresh reduction and accepted common scale

The transaction starts after injection, directory preparation and the first
particle pressure half-step. Candidate accumulation is fused into the existing
post-sampling correction traversals, immediately after each receiving particle
reaches its final corrected state. The selected-stuck subset is deferred to its
existing finalization loop; unselected stuck particles are counted in the mobile
correction full scan. Acceptance follows both kernel completions and precedes
particle relaxation. Thus every receiver, including new injections and changed
velocities, is evaluated in the same epoch used by publication.
Existing couplingOld caches remain frozen as in the original source model.
Neither intervening primitive recovery nor the particle pressure/collision
stages refreshes those caches. Particle pressure and collision kernels do not
consume gas thermal energy; the paired gas settlement and recovery finish before
tracking, the second pressure half-step and gas boundary finalization.

For ordinary particles, calculate the original frozen-gas exponential temperature
candidate, bounded to configured material limits, then calculate its double
enthalpy increment q_i/m_i. Use the secant Cp of that bounded temperature interval
in the reduction, not Cp at its old endpoint:

- A = sum(m_i Cp_sec,i a_effective,i)
- C = sum(m_i Cp_sec,i)
- Q = sum(q_i)
- a_bar = A/C; r = 1+C/Cg
- beta = [1-(1-a_bar)^r]/(r a_bar)

The calculation uses log1p/expm1 and analytic small-interval branch derivatives.
At zero interval it uses the derivative limit; at a_bar=0 or1 it uses exact
endpoint formulas. It never forms a cancellation-prone difference of large H
values to recover a tiny heat increment. The secant makes Q=A(Tg-weighted Ti),
so the ordinary gas temperature is a convex combination. Endpoint Cp weights
failed this property (the frozen regression cooled 4000 K gas below its 2800 K
receiver to2786.395 K). For constant Cp/equal rates this retains the original
finite-capacity exponential mean exactly. Heterogeneous/nonlinear finite steps
are an explicitly bounded approximation with measured first-order convergence.
Zero net Q does not suppress opposite-direction particle relaxation.

A final shared scale bounds positive gas expenditure by the current authoritative
gas E above the actual stored-momentum kinetic energy plus existing temperature
floor. It does not add heat or create a new rejection/trap. Existing relaxation
passes recompute their candidate from the same epoch, publish scaled heat and
carry, and reduce the actual accepted particle inventory change. Gas then receives
the opposite change. Its final double subtraction rounding remains measurable.

## Cold interfaces

1D retains its existing uniform mean-enthalpy frozen-gas exponential, using the
cold secant Cp. Its gas publication occurs once in lane0 before the existing
wall/conduction solve; the latter's old duplicate gas input is disabled. A
positive-node bound is imposed on the proposed uniform increment before reduction.

2D retains its top-surface gas boundary condition and explicit conduction
substeps. It is not replaced by mean-temperature or uniform-profile relaxation.
The fresh reduction reserves the conservative gross positive bound G dt Tg,
with a native-arithmetic roundoff guard. No uncomputed negative 2D heat is credited.
The shared coefficient scales its conductance; the existing substeps integrate
signed gas work in double. Profile publication closes against actual rounded wall
ledger energy, and the gas budget receives actual represented profile change plus
that wall energy. This can undercouple a stiff finite contact step; time refinement
and the preserved spatial model are more important than pretending it is exact.
A tighter initial-node-temperature bound is unsafe across the old sharp Cp
transition. Wall-only/no-gas operation never touches a null gas budget.

## Lifetime, allocation and launch cost

Let N be cell count and P particle capacity. Incremental device storage:

- CHT32 gas rhoE/rhoENext: 2*(8-4)N =8N bytes. CHT64:0. These are the
  authoritative current/RK stage arrays; no additional gas backup/RK arrays.
- CHT paired scratch A/Q:16N bytes only when particle temperature and gas heat
  are enabled. It is not allocated in the disabled path.
- CHT live/compact residual:16P bytes, retained even when gas heat is disabled
  because restart, radiation and wall exchange can use it.
- FP32 lazy radiation affine payload:8N additional bytes;64-bit payload unchanged.
- Host checkpoint download/parse may hold an additional8P-byte carry vector;
  schema8 appends8 bytes per saved particle. This is not another device buffer.
- CHMT/gasUGKP already-double gas authority:0 added persistent energy bytes;
  their particle states have no carry allocation.

With active heat and no radiation payload, the total extra persistent device
storage is24N+16P bytes for CHT32 and16N+16P bytes for CHT64. Lazy CHT32
radiation increases the former to32N+16P. Disabled heat retains only the gas
authority and carry costs (8N+16P for32,16P for64).

The coefficient C reuses rhoENext only after the completed final gas RK stage;
it is fully cleared even in empty cells. Current strictAdvance has no later
rhoENext reader, and every subsequent RK call overwrites it before reading.
Euler does not read it. This is scratch reuse, not a rollback backup; current CHT
strictAdvance has no transactional retry. The typed gas/chemistry backup tests
remain separate. Future retries must not treat this scratch as saved gas energy.

Active heat clears A/Q/C inside the existing cell pool-clear kernel and reduces
candidates inside the existing mobile and selected-stuck correction traversals.
The former standalone heat-clear and candidate-reduction kernels are removed.
Only acceptance and settlement retain separate cell launches, because their
inputs require completed global reductions/publications. Publication is fused
into the existing ordinary/wall-bound relaxation, lane0 of existing1D groups, and
existing2D substeps. These two cell launches replace the former Eulerian
material-heat kernel and standalone primitive recovery: **zero net added
traversals or launches**. Settlement calls recoverGasPrimitiveCell directly and
preserves its gas outputs bitwise without rebuilding obsolete particle moments.

CHT correction kernels have compile-time heat-enabled/disabled specializations;
the disabled specialization retains the original correction body. Host dispatch
selects exactly one launch for each existing correction stage. Disabled heat has
no thermal candidate work or A/Q allocation. This protects its register footprint
from the enabled candidate calculation; final compiler evidence must verify the
result. See pass-fusion.md for the exact partition proof and numerical tests.

The exact source-level scratch traffic, excluding compiler caching and atomics'
hardware implementation, is: clear writes24N bytes; accept reads24N and writes16N;
settlement reads8N budget bytes. Thus the new transaction issues72N bytes of
logical scratch reads/writes per active step. Each eligible reduction issues three
double atomic additions (24 bytes updated, or48 bytes for a nominal read+write);
each publishing particle issues one double atomic addition (8 bytes updated,
nominal16 bytes read+write). Publication reads an8-byte accepted coefficient.
Carry adds8-byte reads to candidate/effective-enthalpy evaluation and an8-byte
write to each actual thermal publication; copying/compacting it costs one8-byte
read plus one8-byte write per copied particle. Existing cold profile node widths
do not change. Each CHT32 gas-energy read/write gains4 bytes; CHT64 gains none.

The fused ordinary candidate calculation also reads existing mass, diameter, temperature,
three velocity components, status/cell/wall tags and five old gas-cache scalars.
Stuck paths omit mobile velocity reads and inspect existing profile nodes (8 or64
floats);1D additionally scans the same eight nodes for its admissible bound.
These are genuine extra input reads, not covered by the scratch-byte total.
Their physical DRAM traffic depends on particle eligibility, compiler load reuse
and cache behavior. There is no honest universal measured byte count without a
GPU run. Arithmetic adds double caloric inversion, stable enthalpy increments,
sqrt/expm1 and cell log1p/expm1 evaluation; this is not a free pointer-only change.

Node storage remains float. Directed float node publication bounds carry by one
storage ULP while preserving positive effective nodes; the pre-acceptance cooling
margin covers double cancellation and adds no node/global traversal. Cold scratch becomes double only for CHT's carry
path;8-node/64-node storage is not promoted. Native register/stack counts require
the final compiled snapshot and are not inferred from HOST tests. Finite wall
inversion uses64 double bisections versus48 native bisections; radiation retains
its existing validation/apply passes. Earlier chemistry stack measurements remain
in the energy migration cost evidence and must be rechecked on the integrated
source. No GPU timing, atomic-reduction result or runtime validation is claimed.

## Verification and evidence status

The tests execute production carry and source bodies, actual eight-lane cold
solvers, real lifecycle/codec functions, wall/radiation publication and exact
old/new EOS/RK/source regressions. The native fixture calls the complete actual
resident strictAdvance, with real mesh allocation, boundaries, directory,
collision, tracking, compaction and finalization. It is compile/run opt-in and
runtime remains NOT_RUN without a GPU. HOST sequential atomic adapters do not
establish real GPU reduction ordering behavior.

Signed0.02/0.03/0.05 J and100,000 signed prototype updates survive. Production
particle-first cumulative tests use a summed per-step publication-roundoff bound,
not relative error against a large background energy. They report actual error,
absolute exchanged heat and their ratio. Independent long-double RK4 caloric ODE
integration verifies time refinement. Stiff unequal masses/rates are advanced to
represented equilibrium, and no-heat/dt0 state identity is exact. Final measured
numbers and native source hashes belong in the result record after freeze.
