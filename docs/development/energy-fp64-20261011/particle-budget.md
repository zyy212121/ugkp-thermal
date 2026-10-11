> Historical evidence note (updated 2026-10-11): blocked-migration and old
> coupling conclusions below describe the pinned earlier snapshot, not the current
> paired-carry implementation. The current design/cost contract is
> [paired-carry-design.md](paired-carry-design.md); current verification and
> remaining native-runtime limits are in [current-verification.md](current-verification.md).

# Actual particle publication: local-kernel budget and design scope

This HOST-only probe invokes the unmodified production mobile-particle relaxation,
Eulerian drag source, and Eulerian material-heat source. It reads the published
particle velocity, unresolved theta and temperature, and gas mass, momentum and
energy, before and after each invocation. It then executes the production atomic
particle-moment accumulation and normalization to inspect the published material
enthalpy moment. CHT has no per-particle `pH` array: material enthalpy is derived
from `pT` and published as `momRhoHpP`.

The fixture supplies a linear material constitutive relation (`cp = 1000`,
`h = 1000 T`, `T = h / 1000`) and a prescribed drag inverse time of one. Those
are explicit input adapters, not replacements for exchange updates. Actual
production heat and momentum relaxation expressions and writes are compiled.
One mobile parcel occupies a synthetic unit-volume compartment; no complete
mesh transport or closed-boundary solver step is executed. Gravity and pressure
forces are zero; wall interactions are inactive. Wall-only supporting types are
real production headers; the three unavailable wall side-effect adapters abort
if accidentally reached. No production source is modified by this test.

Independent Python Decimal budgets use the published fields, not the intended
`dHp`, `thetaDragAlpha` loss, or reconstructed intended particle velocity.
Mass, all three momentum components, mechanical energy, and material enthalpy
are compared. Double analytical estimates diagnose a difference between the two local
kernel models present in both the baseline and current source: particle relaxation freezes old gas velocity/temperature,
whereas gas sources use finite-capacity coupled relaxation. Separately record
input/moment/particle quantization and residuals after every actual publication.
These diagnostics must not be presented as conservation/pairing fixes.

Cases cover both Real widths with authoritative double gas energy: zero-time,
strict zero exchange, finite drag with and without large kinetic background,
unresolved-theta-only transfer, finite heat, and sub-ULP particle temperature
changes. Optional baseline runs compile the same actual-particle harness against
an explicitly supplied exact source snapshot, using its original homogeneous
energy storage. They identify residuals of this local kernel composition already present in
the baseline, rather than assuming all nonzero totals were introduced by this
precision change. They do not establish a complete production-advance defect. No CUDA execution,
device reduction, complete resident transaction, or full solver validation is
claimed.

## Results

All fourteen current-source cases pass; both optional exact-baseline cases pass
(16 tests total). These include explicit characterization tests requiring nonzero
residuals. Their success is **not conservation acceptance**. No existing test
threshold was changed.

Run on 2026-10-11 with g++ HOST serial emulation, both Real widths:

```
PYTHONPATH=/workspace/scratch/98b6339ec81b/chmt_implementation_20261011/python_deps \
PYTHONDONTWRITEBYTECODE=1 UGKWP_PARTICLE_BUDGET_BASELINE_ROOT=/tmp/energy-base \
python -m pytest -p no:cacheprovider tests/gas_transport/test_actual_particle_energy_budget.py -q -s
```

The baseline is `ab900f54` extracted into `/tmp/energy-base`, with only the
already-documented include-order fix. Its original gas energy precision is used:
float for bits32, double for bits64. Current bits32 uses double gas energy and
float remaining fields. All before/after particle fields and reaccumulated
particle moments are also compared directly between baseline and current source.

The budgets use exact binary values converted to 70-digit Decimal. The physical
particle-state energy is `epsG*rhoE + m*(u²/2 + 1.5*theta + 1000*T)`. The second,
cell-moment publication budget is `epsG*rhoE + momRhoEP + momRhoHpP`. They differ
when the production Real moment reduction rounds a stored particle's energy.
Neither substitutes an intended exchange for the actual published particle.

Gas initial energy is consistent with conserved momentum and `Tgas`:
`rhoE = |rhoU|²/(2*rho) + rho*R*Tgas/(gamma-1)`, rounded only into its actual
storage type. The high-speed primitive velocity cache and conserved momentum
can still disagree by their native multiplication/division rounding; that is
measured below rather than silently treating the cache as exact.

Common inputs are a 1 m³ cell, one 100 kg represented parcel, solid density
1000 kg/m³, diameter Real(0.001) m, linear cp=1000 J/(kg K), no body/pressure
forces, no active wall interactions. Actual `accumulateParticleMomentsAtomicKernel`
and `normalizeParticleMomentsAtomicKernel` build both initial and final moments.
Thus mass/momentum/energy moment inputs derive from exactly these particles;
there is no independently prescribed cell momentum or energy.

## Summary of actual local-composition total residuals

Energy residuals are joules, momentum residuals kg m/s, mass residuals kg.
Each value is after the fixture's gas publication, actual particle relaxation
writes, and subsequent actual moment accumulation, not after a complete CHT
advance. The fixture omits the intervening and subsequent stages listed below. `Particle total` uses physical energy
from stored particle fields; `Moment total` uses their published cell moments.


| Real bits | Source | Case | Particle total ΔE | Moment total ΔE | Δmass | ΔPx | ΔPy | ΔPz |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 32 | baseline | zero time | 30198988 | 30198988 | 0 | 1.7999999523162842 | 0 | 0 |
| 32 | baseline | zero exchange | 0 | 0 | 0 | 0 | 0 | 0 |
| 32 | baseline | finite drag | -223.90102190428058 | -223.90102007053792 | 0 | 127.3676834689732 | -84.911796855075863 | 53.069876041476746 |
| 32 | baseline | high kinetic drag | 48643645438.292221 | 49274683452 | 0 | 2674.0000033378601 | -84.558505412503891 | 52.849068903280305 |
| 32 | baseline | theta only | -0.011804079258581623 | -0.011804079636931419 | 0 | 0 | 0 | 0 |
| 32 | baseline | finite heat | 714570.71628570557 | 714572.00339508057 | 0 | 0 | 0 | 0 |
| 32 | baseline | micro heat | 0 | 0 | 0 | 0 | 0 | 0 |
| 32 | current | zero time | 0 | 0 | 0 | 0 | 0 | 0 |
| 32 | current | zero exchange | 0 | 0 | 0 | 0 | 0 | 0 |
| 32 | current | finite drag | -223.90783972941898 | -223.90783789567632 | 0 | 127.3676834689732 | -84.911796855075863 | 53.069876041476746 |
| 32 | current | high kinetic drag | 48627135225.379593 | 49258173239.087372 | 0 | 2674.0000033378601 | -84.558505412503891 | 52.849068903280305 |
| 32 | current | theta only | 4.3817749428853858e-10 | 5.9827698351000436e-11 | 0 | 0 | 0 | 0 |
| 32 | current | finite heat | 714565.75195307774 | 714567.03906245274 | 0 | 0 | 0 | 0 |
| 32 | current | micro heat | -0.034439991992314845 | -0.034439991992314845 | 0 | 0 | 0 | 0 |
| 64 | baseline | zero time | 0 | 0 | 0 | 0 | 0 | 0 |
| 64 | baseline | zero exchange | 0 | 0 | 0 | 0 | 0 | 0 |
| 64 | baseline | finite drag | -223.90786559296248 | -223.90786559296251 | 0 | 127.36769665864732 | -84.911797772431527 | 53.069873607769701 |
| 64 | baseline | high kinetic drag | 49197173290.163445 | 49197173290.256248 | 0 | 2705.87220665738 | -84.558506456588333 | 52.849066535367704 |
| 64 | baseline | theta only | 1.0387031614172378e-11 | 1.0387031803907758e-11 | 0 | 0 | 0 | 0 |
| 64 | baseline | finite heat | 714565.14627300482 | 714565.14627300506 | 0 | 0 | 0 | 0 |
| 64 | baseline | micro heat | 7.8132870831411814e-09 | 6.7055225363817548e-09 | 0 | 0 | 0 | 0 |
| 64 | current | zero time | 0 | 0 | 0 | 0 | 0 | 0 |
| 64 | current | zero exchange | 0 | 0 | 0 | 0 | 0 | 0 |
| 64 | current | finite drag | -223.90786559296248 | -223.90786559296251 | 0 | 127.36769665864732 | -84.911797772431527 | 53.069873607769701 |
| 64 | current | high kinetic drag | 49197173290.191574 | 49197173290.284378 | 0 | 2705.87220665738 | -84.558506456588333 | 52.849066535367704 |
| 64 | current | theta only | 1.0387031614172378e-11 | 1.0387031803907758e-11 | 0 | 0 | 0 | 0 |
| 64 | current | finite heat | 714565.14627300482 | 714565.14627300506 | 0 | 0 | 0 | 0 |
| 64 | current | micro heat | 7.8132870831411814e-09 | 6.7055225363817548e-09 | 0 | 0 | 0 | 0 |

## High-kinetic drag: complete local-composition published-state ledger

This case has dt=0.25 s and prescribed beta=1 s⁻¹. Nominal gas velocity is
`(Real(19999920)/Real(1.1), -3, 2)` m/s; parcel velocity is the gas x cache minus
128 m/s, with y=1 and z=-0.5 m/s. Initial theta is Real(0.0002) (m/s)² and both
temperatures are 300 K. Gas density is Real(1.1). The represented solid fraction
is Real(100/1000); its rounded gas complement is retained in all budget totals.
Initial unresolved particle energy is present in actual `pTheta`, although it
cannot survive adding it to a 1.65e16 J FP32 mechanical cell moment. The actual
moment rounding, rather than manually setting a zero-theta input, is exercised.

The following full current-source ledger separates gas kinetic/internal energy,
particle resolved/unresolved energy, and material enthalpy. Values are calculated
from stored fields. Baseline gas-energy update differences follow afterward.

### Current Real bits=32

| Quantity | Before | After | Change |
|---|---:|---:|---:|
| gas_mass | 0.98999999523162785 | 0.98999999523162785 | 0 |
| particle_mass | 100 | 100 | 0 |
| mass | 100.98999999523163 | 100.98999999523163 | 0 |
| x | 1836161729.3231647 | 1836164403.323168 | 2674.0000033378601 |
| y | 97.029999907016759 | 12.471494494512868 | -84.558505412503891 |
| z | -48.020000009536744 | 4.8290688937435604 | 52.849068903280305 |
| gas_kinetic | 163635079393678.34 | 163632788501644.62 | -2290892033.7062969 |
| gas_internal | 213097.51310158666 | 9670818.3067698553 | 9457720.7936682682 |
| gas_energy | 163635079606775.84 | 163632798172462.94 | -2281434312.9126287 |
| particle_resolved | 16528561654896262 | 16528612563465800 | 50908569538.304031 |
| particle_unresolved | 0.029999999242136255 | 0.018195919983554631 | -0.011804079258581623 |
| enthalpy | 30000000 | 30000000 | 0 |
| energy | 16692196764503038 | 16692245391638264 | 48627135225.379593 |
| published_energy | 16692196125488504 | 16692245383661742 | 49258173239.087372 |
| published_x | 1836161721.3231647 | 1836164411.323168 | 2690.0000033378601 |
| published_y | 97.029999907016759 | 12.471494494512868 | -84.558505412503891 |
| published_z | -48.020000009536744 | 4.8290688937435604 | 52.849068903280305 |

Actual storage values:

| Field | Before | After |
|---|---:|---:|
| eps | 0.89999997615814209 | 0.89999997615814209 |
| rho | 1.1000000238418579 | 1.1000000238418579 |
| gx | 19999922 | 19999782 |
| gy | -3.3000001907348633 | 1.0568670034408569 |
| gz | 2.2000000476837158 | -0.52304190397262573 |
| ge | 181816759935139.12 | 181814225008057.62 |
| mass | 100 | 100 |
| px | 18181618 | 18181646 |
| py | 1 | 0.1152031421661377 |
| pz | -0.5 | 0.052998065948486328 |
| theta | 0.00019999999494757503 | 0.00012130613322369754 |
| temperature | 300 | 300 |
| published_h | 30000000 | 30000000 |
| published_mechanical | 16528561015881728 | 16528612555489280 |
| published_px | 1818161792 | 1818164608 |
| published_py | 100 | 11.52031421661377 |
| published_pz | -50 | 5.2998065948486328 |
| gas_ux_cache | 18181746 | 18181746 |
| gas_uy_cache | -3 | -3 |
| gas_uz_cache | 2 | 2 |

Current minus baseline actual ΔE: -16510212.912628174 J. All actual particle and moment publications are identical between source versions; this difference is on the gas update/storage side.

### Current Real bits=64

| Quantity | Before | After | Change |
|---|---:|---:|---:|
| gas_mass | 0.9900000000000001 | 0.9900000000000001 | 0 |
| particle_mass | 100 | 100 | 0 |
| mass | 100.98999999999999 | 100.98999999999999 | 0 |
| x | 1836161673.4545453 | 1836164379.3267519 | 2705.87220665738 |
| y | 97.030000000000001 | 12.471493543411666 | -84.558506456588333 |
| z | -48.020000000000003 | 4.8290665353677031 | 52.849066535367704 |
| gas_kinetic | 163635054548079.16 | 163632773151149.41 | -2281396929.737061 |
| gas_internal | 213097.47281473858 | 221139.4942509316 | 8042.0214361930357 |
| gas_energy | 163635054761176.62 | 163632773372288.91 | -2281388887.7156253 |
| particle_resolved | 16528560663171656 | 16528612141733834 | 51478562177.918999 |
| particle_unresolved | 0.030000000000000002 | 0.018195919791379005 | -0.011804080208620997 |
| enthalpy | 30000000 | 30000000 | 0 |
| energy | 16692195747932834 | 16692244945106124 | 49197173290.191574 |
| published_energy | 16692195747932832 | 16692244945106122 | 49197173290.284378 |
| published_x | 1836161673.4545453 | 1836164379.3267519 | 2705.8722067020835 |
| published_y | 97.030000000000001 | 12.471493543411666 | -84.558506456588333 |
| published_z | -48.020000000000003 | 4.8290665353677031 | 52.849066535367704 |

Actual storage values:

| Field | Before | After |
|---|---:|---:|
| eps | 0.90000000000000002 | 0.90000000000000002 |
| rho | 1.1000000000000001 | 1.1000000000000001 |
| gx | 19999920 | 19999780.580255471 |
| gy | -3.3000000000000003 | 1.0568670164996834 |
| gz | 2.2000000000000002 | -0.52304188531230222 |
| ge | 181816727512418.47 | 181814192635876.56 |
| mass | 100 | 100 |
| px | 18181617.454545453 | 18181645.768045221 |
| py | 1 | 0.11520313228561951 |
| pz | -0.5 | 0.052998042321487748 |
| theta | 0.00020000000000000001 | 0.0001213061319425267 |
| temperature | 300 | 300 |
| published_h | 30000000 | 30000000 |
| published_mechanical | 16528560663171656 | 16528612141733834 |
| published_px | 1818161745.4545453 | 1818164576.804522 |
| published_py | 100 | 11.520313228561951 |
| published_pz | -50 | 5.2998042321487748 |
| gas_ux_cache | 18181745.454545453 | 18181745.454545453 |
| gas_uy_cache | -3 | -3 |
| gas_uz_cache | 2 | 2 |

Current minus baseline actual ΔE: 0.02812957763671875 J. All actual particle and moment publications are identical between source versions; this difference is on the gas update/storage side.

## Interpretation and strict limits

1. **Baseline local-kernel model difference.** Gas drag advances a finite-capacity pair,
   decaying relative velocity with `exp(-dt*beta*(1+ms/mg))`. The actual particle
   freezes the old gas velocity and uses `exp(-dt*beta)`. The cell source's
   intended solid velocity therefore does not equal the velocity it actually
   publishes in this local composition. The FP64 local probe retains the large
   residual, and the exact baseline reproduces it. This establishes different
   local kernel exchange models, not a measured full production-advance defect.
   No claim is made that this work fixes pairing.
2. **Input and publication quantization.** For the high-speed mixed32 case the
   independent continuum model predicts about 4.95349e10 J residual. Actual
   cache-vs-conserved-input differences contribute about -3.53372e8 J; rounding
   the actual particle velocity publication contributes about -5.69994e8 J.
   The remaining gas/moment arithmetic difference is about +1.56151e7 J. These
   are model-diagnosis quantities, not conserved-error tolerances. Separately,
   reaccumulating FP32 particle moments changes the apparent budget by hundreds
   of millions of joules. The actual particle-field ledger remains primary.
3. **New exposed micro-heat budget failure.** At dt=1e-8 s, gas/particle velocities
   are both `(40000,0,0)` m/s, gas temperature 500 K, particle temperature 300 K,
   viscosity Real(1e-5), and heat is active with drag disabled. Double gas
   authority retains a roughly 0.03444 J loss; actual float pT remains exactly
   300 and actual momRhoHpP remains exactly 30000000. The resulting mixed32
   total residual is **-0.034439991992314845 J**. Baseline float gas authority
   discarded the same loss and reported zero. This is a real full-budget
   failure of the tested local composition exposed by retaining the gas
   exchange, and blocks claiming that the gas-authority change proves actual
   particle pairing/conservation. A complete advance still requires separate
   end-to-end validation. The characterization test requires that nonzero
   residual; it does not accept it as conservation using a widened tolerance.
4. **Baseline finite-heat local-model difference.** At dt=0.25 s, gas uses finite
   capacities while particle temperature freezes old gas temperature. A roughly
   +714565 J local-composition residual survives FP64 and baseline runs. It is
   not a measured complete CHT step residual or proof of an existing production
   solver bug. Linear constitutive input adapters isolate the local model
   difference but do not instantiate the production alumina constitutive law.
5. **Validated precision-only improvements.** Zero-time and strict-zero-exchange
   cases preserve all actual conserved/particle publications in both current
   widths. The theta-only case measures the actual pTheta loss and gas gain;
   current residual is sub-nanojoule rather than losing the gas-side transfer.

A physical full-pairing fix requires a separately reviewed source/particle
contract. This evidence adds no trap, refusal, compensation, source-model change,
or production code. It does not claim GPU execution or full resident ordering.

## Baseline high-kinetic input quantization caveat

The nominal conservative initialization is physically consistent before storage,
but **the exact old bits32 storage cannot represent its positive internal energy**.
Its stored gas internal energy, independently calculated from stored rhoE and
momentum, is -5,677,916.3433398366 J per cell. The current double authority retains
positive internal energy; current/FP64 initial gas internal energy is approximately
213,097.47 J per cell. The old bits32 source's existing repair/arithmetic therefore
also contributes to the baseline/current update difference. The -16,510,212.912628174 J
difference in their high-kinetic residuals is not an isolated valid-input source
regression measure. The FP64 baseline and current initial states are valid and
independently establish the baseline 4.9197e10 J residual of this local kernel
composition, not an end-to-end production-advance residual. This caveat
must accompany any quotation of the mixed32 high-kinetic baseline comparison.

## Read-only design audit: pairing actual material exchange

This section is a proposal/investigation only. No production implementation,
exchange-model change, new failure policy, or gas-energy compensation was made.
Tests above were frozen before this audit.

### Verified current ordering

In `applications/CHT/gpu/GpuResidentStrict.cu`:

1. Around lines 4845–4948, gas volume-fraction source and solid recovery run,
   then Eulerian drag, then `applyEulerianParticleMaterialHeatKernel`; gas/solid
   recovery follows the source stages. The heat kernel immediately subtracts its
   finite-capacity estimate from gas rhoE (common kernel lines 126–151).
2. Injection, pre-transport directory preparation, and the first collisional
   pressure half-step then run. The collision pool samples/corrects particle
   velocity and theta. Those operations can change the particle Reynolds number
   later used for material heat; the inspected pool paths do not rewrite pT.
3. Around lines 5116–5212, cold-wall 1D, mobile relaxation, wall-bound relaxation,
   and cold-wall 2D execute. Ordinary mobile/wall-bound gas heat writes pT inside
   `common/GpuThermalParticleRelaxation.cuh:127–140`. Finite-contact wall heat
   may write pT again afterward (`FiniteContactRelaxation.inl:93`). Cold-wall
   paths have separate gas/wall/phase-change state and must not be omitted.
4. Packing and tracking then run. Tracking can move, remove, or change the wall
   state of particles. Only afterward, around line 5260, does
   `launchPostTransportMomentPipeline` clear/reaccumulate particle moments and
   recover solid fields. Its implementation is `common/GpuMomentPipeline.cuh`.
5. The special material-enthalpy refresh around lines 5900–5968 is a separate
   post-radiation API. It is not a free reduction already performed at the gas
   relaxation boundary. Calling it here would add traversal(s).

Consequently, subtracting old and new `momRhoHpP` after the existing end-of-step
reduction does **not** isolate gas-to-particle heat. It would include injected or
removed parcels, transport between cells, and wall/phase-change heat. It would
also subtract two large Real moments, reintroducing the precision loss. The
current frozen coupling caches preserve gas input, but there is no existing
stored per-particle old material enthalpy or gas-only enthalpy increment.
`rhoHp` and `momRhoHpP` are active Real solid-state fields; they cannot simply be
repurposed as double scratch buffers. Reusing rhoENext or pressure/collision
scratch requires a separate lifetime/aliasing audit and is not established here.

### Candidate A: measure actual gas-only particle writes in the existing pass

A narrowly scoped new contract could read the old stored pT, run the existing
particle gas-heat write, then measure the actual new stored pT before wall heat.
It would accumulate `mass*(h(newStoredT)-h(oldStoredT))` in a double per-cell
exchange buffer keyed to the source cell **before tracking**. This captures zero
exchange when float pT does not move; no intended dHp is substituted for it.
Thermodynamic evaluation must use the same documented authoritative material
energy definition on both sides, rather than subtracting rounded large Real
cell moments. The 1D/2D cold-wall branches require equivalent gas-only accounting,
separate from wall/phase-change energy.

- Added persistent allocation: 8 bytes × nCells for the FP64 exchange sum.
  If gas-volume normalization is frozen at the original source boundary, another
  8 bytes × nCells for its double snapshot is the conservative design estimate.
  Whether the denominator can be safely reconstructed later is not established.
- Added per-particle live arithmetic only: old/new enthalpy evaluation plus a
  cell FP64 reduction/atomic contribution. No new particle sweep is intrinsically
  required for ordinary mobile relaxation. Double atomics introduce contention
  and order-dependent rounding; a deterministic segmented reduction would need
  a different relaxation/reduction organization or explicit per-particle deltas.
- Reset can be fused into an existing pre-relaxation cell kernel. Final gas
  publication must happen after every participating relaxation kernel completes.
  An explicit publication/recovery kernel adds one O(nCells) sweep; fusing into
  an already existing suitable cell stage might avoid that sweep, but requires
  a verified dependency/lifetime audit. **Zero added global sweeps is not yet
  demonstrated.** No claim of zero performance cost is made.
- The early finite-capacity gas subtraction must be replaced/deferred, not kept
  and silently followed by an unreviewed correction. This moves gas publication
  and EOS recovery past injection/pressure/collision processing, and must be
  reviewed for all intermediate consumers and boundary/diagnostic timing.
- Even if this gives exact pairing to stored pT, repeated sub-ULP heat steps can
  leave pT and both phase energies unchanged forever. It conserves the stored
  budget by discarding unresolved transfer. It does **not** satisfy persistent
  accumulation of physically nonzero sub-ULP material exchange.
- It also replaces the gas finite-capacity heat contract with the actual
  frozen-gas particle result. Strong exchange can exhaust gas internal energy;
  an independently floored gas update would again break pairing. A shared
  admissible exchange/limiting contract is required before implementation.

Thus candidate A offers a plausible no-extra-**particle**-pass accounting path,
but it is not a precision-only fix and is not approved for production here.

### Candidate B: persistent double particle material-energy authority

To retain micro-heat exchange while pT remains float, a particle must carry that
energy somewhere authoritative. One design is double specific enthalpy or
represented material energy, with pT becoming a derived float cache. The actual
shared exchange updates gas rhoE and this particle authority with opposite signs.
Gas finite capacity versus particle frozen-gas relaxation still needs a reviewed
shared source contract; adding storage alone cannot reconcile the two models.

- New live particle double field: 8 bytes × particleCapacity.
- Corresponding compact/gather target: another 8 bytes × particleCapacity,
  because this implementation swaps live/compact particle buffers. Added total
  is at least **16 bytes × particleCapacity**, while existing pT caches remain.
- A dedicated double cell exchange accumulator adds 8 bytes × nCells; a frozen
  gas-volume snapshot adds a further 8 bytes × nCells if required by the chosen
  ordering. Transaction/rollback snapshots, if introduced, cost additional
  storage and are not included in this lower bound.
- Initialization, injection, restart/checkpoint, compaction/gather, wall contact,
  cold-wall 1D/2D, radiation, material moments and downloads must all transport
  and update the authority. Each compacting pass adds at least one 8-byte read
  plus one 8-byte write per surviving parcel for the new field; updates likewise
  add an authority load/store. Actual traffic and cache effects require profiling.
- Existing passes can potentially carry this extra field without adding a new
  full particle sweep. A double material-thermo evaluation/inversion path and
  double-capable moment reduction are needed so the new authority is not
  immediately rounded back through float h(T). Persistent sub-ULP history must
  survive every lifecycle operation, not merely one relaxation kernel.

This is a larger physical-state/lifecycle change and cannot be slipped into the
current gas-energy scalar patch. Neither candidate addresses the independent
resolved/unresolved drag local-pairing discrepancy demonstrated above. No production
code or tests were changed while investigating these options.


## Caller, timestep and omitted-stage verification (scope correction)

This verification was requested before attributing any finite-step local result
to a production solver bug. It is a read-only source audit; tests remain frozen.
**No complete CHT advance was executed by this probe.** The +714565 J finite-heat
and finite-drag momentum/energy numbers must be quoted only as local actual-kernel
pairing characterizations with the stated synthetic inputs.

### Real timestep chain and both half-steps

- `applications/CHT/diluteUgkwpFoam.C:853–857` invokes
  `resident.advanceOneStep(runTime.deltaTValue(), runTime.value())` once after
  incrementing run time. `GpuResidentStrict.H:2064–2074` forwards dt unchanged
  to `ugkwpGpuResidentStrictAdvance`; `GpuPrecisionDispatch.C:1062–1069` selects
  32/64 backend without changing dt. There is no caller half-step around the
  entire particle/gas heat or drag update in this inspected CHT path.
- In `GpuResidentStrict.cu:4878–4890`, Eulerian drag receives the full dt once;
  the material heat launch at line 4927 likewise receives full dt once.
  The mobile relaxation launches at lines 5148–5164 and wall-bound alternatives
  at 5174–5190 each receive full dt for their eligible, separate populations.
  The fixture's source(dt), then mobile relaxation(dt), therefore matches these
  individual calls' dt values. They are not dt/2 calls.
- `applyCollisionalPressureKick<true>` at line 4973 uses `0.5*dt` before collision
  sampling and particle relaxation. `applyCollisionalPressureKick<false>` at
  line 5269 uses `0.5*dt` after tracking and post-transport moment recovery.
  The fixture executes **neither** pressure half-step. These are pressure
  projections, not a second invocation or reweighting of the gas heat/drag pair.
  Their source dispatch is `common/GpuPressurePipeline.cuh` through the local
  wrapper at `GpuResidentStrict.cu:2295–2300`.
- `advanceGasFluxStage` occurs earlier at line 4809. In
  `common/GpuGasAdvance.cuh:334–384`, Euler/RK2/RK3 have their own flux-stage
  substeps and 1/2–1/2, 3/4–1/4, or 1/3–2/3 conservative blends. These return
  before the particle coupling sequence. They do not wrap or blend the later
  Eulerian heat/drag and particle relaxation calls. The two chemistry dt/2
  calls in the generic trial helpers are not calls on this inspected CHT strict
  advance path and cannot be silently counted as particle coupling half-steps.
- The exact baseline snapshot has the same relevant callsites/order and full-dt
  versus half-dt structure. This rules out an accidentally doubled local dt in
  the fixture relative to the inspected callsites; it does not make the omitted
  stages or constitutive adapters equivalent to a complete production step.

### What the fixture executes and omits

The fixture initializes actual particle arrays and production moments, executes
one Eulerian drag **or** material-heat kernel, executes
`relaxOneParticleToResidentGas`, then immediately reaccumulates/normalizes moments.
It passes constant beta=1 input adapters and a linear cp/h/T constitutive adapter.
The production Schiller–Naumann/Gidaspow drag and alumina constitutive models are
not instantiated. Cases 0–4 disable particle material heat; cases 5–6 disable
drag and have equal initial gas/particle velocities. Therefore their local source
is intentionally isolated, not a replay of the whole resident state machine.

Omitted stages and their relevance are:

1. **Gas fluxes, diffusion/turbulence, gas boundary work and boundary finalization.**
   No full geometry/boundary transport problem is executed. Their contributions
   cannot be presumed zero from the local ledger alone. The synthetic dt=0.25 s
   at roughly 1.8e7 m/s also has not been shown to satisfy a production mesh CFL,
   tracking-hop, or timestep constraint. It is a numerical source stress case.
2. **Gas gravity and pressure-force inputs.** The fixture explicitly sets gravity
   and gas pressure gradients to zero. Production gravity would run before the
   source and pressure gradients are computed before drag. These explicit local
   inputs remove their work from this test but do not test their solver plumbing.
3. **Gas volume-fraction source/history.** It is omitted. The fixture has fixed
   mass/volume and initializes epsGPrev to its current gas fraction, so an ideal
   zero-history-change source is intended. Its actual kernel/history lifecycle
   is not exercised by this particle-budget probe.
4. **Primitive/solid recovery after volume, drag, heat and pressure.** These calls
   are omitted. The local particle kernels read saved coupling gas caches, which
   are explicitly populated as in their source/snapshot path; subsequent gas
   primitive recovery does not itself replace those saved old caches. However,
   recovery can repair/round other gas/solid state and provides inputs for the
   stages that the fixture does not execute. No all-stage repair budget is shown.
5. **Injection, directory updates, both pressure half-steps and collision pool.**
   No parcels are injected or selected by a pool in the fixture. Actual pressure
   and pool stages can change particle velocity/theta and thus heat Reynolds
   numbers and moments. The inspected pool sampling/correction launches receive
   `applyThetaDrag=0`; ordinary relaxation subsequently applies thetaDragAlpha.
   These facts do not prove the omitted stages are identities for all synthetic
   cases. Their mechanical budgets and stochastic/constraint effects remain
   outside the probe.
6. **Cold-wall/finite-contact physics, packing and tracking.** They are not
   executed. The tested parcel is mobile, with no wall contact. A real moving
   parcel can cross cells, leave the domain, or acquire wall state; packing and
   pressure projections can alter mechanical energy. A real wall can exchange
   energy/momentum. No boundary ledger closes those possibilities here.
7. **Post-transport recovery and compaction.** The test uses real serial atomic
   moment accumulation/normalization immediately after relaxation. Production
   does so after tracking and may use segmented/heavy reductions, solid recovery,
   a second pressure projection and buffer compaction. The test proves actual
   writes and this accumulation's quantization, not equivalence to those paths.
8. **External CHT/radiation exchange.** After `advanceOneStep`, the application may
   call `solidThermalCoupler->exchangeIfDue` (lines 860 onward). The fixture has
   no such coupling or external energy ledger. These are additional system
   exchanges, not an implicit later half-step of the tested local heat kernel.

### Supported conclusion after review

The exact old and current local kernels, under identical explicit adapters,
produce different intended gas-side and actual particle-side exchanges at finite
dt, and the retained micro-heat gas loss exposes an actual float particle-write
budget gap. The source-call dt is verified, but no cancellation, persistence, or
magnitude of these finite-step residuals has been established across the complete
production advance. The finite-step baseline numbers are therefore **not labeled
an old production bug**. At this unpaired snapshot, the micro-heat result blocked
a claim of actual gas/particle energy pairing. The later bounded thermal carry
and particle-first paired publication supersede that implementation; their
current host evidence is recorded in current-verification.md. The original
0.03444 J observation remains historical causality evidence. Establishing full
solver conservation still requires an end-to-end case with matching
constitutive laws, pressure/collision processing, mesh transport, recovery,
boundary/external ledgers, and timestep admissibility.
