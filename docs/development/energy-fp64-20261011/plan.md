> Historical evidence note (updated 2026-10-11): blocked-migration and old
> coupling conclusions below describe the pinned earlier snapshot, not the current
> paired-carry implementation. The current design/cost contract is
> [paired-carry-design.md](paired-carry-design.md); current verification and
> remaining native-runtime limits are in [current-verification.md](current-verification.md).

# Persistent authoritative FP64 gas energy implementation plan

> Execution uses the approved independent energy branch, native implementation, test-first steps, and an independent review before integration. No README changes, merges, physics changes, or production GPU claims.

Goal: preserve physically small gas-energy increments in CHT's FP32 resident state by making rhoE and its RK copy persistent binary64 energy densities, independent of mass/momentum/particle precision.

Architecture: an independently typed energy member in the shared non-owning gas view, deduced by its binder; CHT owns two double arrays. Existing double owners remain double. Every current-energy reader/writer operates with the energy scalar and narrows only explicitly derived primitive outputs. Thermochemical tables retain their declared storage type while local reactor/EOS arithmetic follows the authoritative energy type.

Tech stack: C++17, CUDA, OpenFOAM 10; pytest-driven host execution of production kernels, native CUDA compile checks. Source design: approved geometry_precision_design.md sections 4 and 7 supplied with this task; the contracts below carry its energy requirements into the repository.

## Global constraints

- rhoE is energy density; CHMT GasQ.energy is already double integrated energy. No claim that promoting CHT fixes preexisting CHMT/gasUGKP precision.
- Keep other state arrays and face-flux arrays in their existing Real type. This fixes inventory/update narrowing, not quantization already present in face fluxes, material tables, species publication, or particle inventories.
- No shadow float energy, pointer reinterpretation, additional reaction heat, new p*dV, relaxed bounds, or new whole-grid conversion passes.
- Existing positivity/repair contracts remain. Valid EOS recovery must not overwrite the energy authority from primitive caches.
- ALE remains Euler-only. Energy update uses explicit verified Vold and Vnew in energy precision; chemistry uses Vold then Vnew from its caller.
- Old float restart values promote their already-quantized bits. No history reconstruction.

## Complete write-set and lifetime inventory

| Owner/path | Current storage or operation | Required treatment |
|---|---|---|
| CHT gpu/GpuResidentStrict.cu DeviceState rhoE/rhoENext | GpuReal pointers; typed allocate/release; POD refresh | two double pointers, typed allocation/copy/free, mirror remains POD pointer copy |
| CHT C ABI, GpuPrecisionDispatch, GpuResidentStrict.H | double host input/output and OpenFOAM scalarField | retain ABI, verify double scalar build, no float round trip |
| CHT createFields.H | READ_IF_PRESENT rhoE density, initial p/(gamma-1)+rho*U²/2 | retain double field and accepted restart value |
| CHT diluteUgkwpFoam.C | upload/advance/download/write and thermal checkpoint calls | prove the downloaded authority reaches write; no reconstruction from p/T |
| CHT thermal/GpuThermalExchangeState.C | hashes written fluid/rhoE, requires binary or max_digits10 ASCII | retain exact round-trip and existing metadata; explicitly record energy width/representation in thermal checkpoint state if compatibility allows |
| common/operators/computeSstFaceFluxKernel.cuh | divergence; save/blend; mixture/ALE/legacy EOS | independent energy accumulator, scale/volume operations; double RK weights; high precision EOS scratch and conditional legacy repair |
| common/GpuGasAdvance.cuh and GpuGasHostPolicy.cuh | RK weights sourced from policy Weight | double weight API/constants before any Real conversion |
| common/GpuGravityUpdate.cuh | current energy and mechanical work | energy-precision accumulator and work; momentum precision unchanged |
| common/GpuCellDragUpdate.inl | gas/particle partition, kinetic/internal energy and dissipation | energy-precision gas energy and transfer budget; original particle fields unchanged and quantization documented |
| common/operators/applyGasVolumeFractionSourceKernel.cuh | phase-volume energy rescale and pressure work | independent energy arithmetic; original guard and mass/momentum source unchanged |
| common/operators/applyEulerianParticleMaterialHeatKernel.cuh | gas decrement from dHp and lower bound | energy-precision authority and update, explicitly preserve published particle-exchange quantization |
| common/operators/computeRiemannGasFaceFluxDevice.cuh | state read into Real conservative flux input | explicit derived flux conversion, no writeback; input-flux quantization remains outside inventory guarantee |
| common/operators/advanceGasChemistryKernel.cuh | rho-derived reactor Real truncates U | energy-typed input/scratch, mixed table view, original species publication and closure check; rhoE read-only |
| common/gasTransport/MixtureThermo.H; common/chemistry/*.H | homogeneous table/scratch signatures | independent TableReal template, no copied per-cell tables; controls/audit conversion at adapter boundary |
| gasUGKP SharedGasTrialPolicy; CHMT accepted/trial/microBase/checkpoint | already-double snapshots and schema | unchanged ownership and checkpoint contract; regression verification |

CHT does not currently have a generic reject/retry gas transaction snapshot: rhoENext is the RK initial state, not an accepted-state checkpoint. Thermal restart persists downloaded OpenFOAM fields. Do not claim generic mixed-owner transactions based on gasUGKP's all-double snapshot implementation.

## Interfaces

- GasStateView<Real, Time, SstCoefficients, Energy=Real>: rhoE/rhoENext are Energy*, EnergyType is exposed; default preserves old three-argument callers.
- makeGasStateView(owner): deduces Energy from owner.rhoE and statically requires matching rhoENext.
- GasEnergyScalar<State>: scalar deduced from rhoE; helpers accept this scalar without passing through GPU_OPERATOR_REAL-only finiteOr/clampMin.
- blendGasConservativeStateKernel(State*, double initialWeight, double stageWeight): nonenergy weights explicitly converted to Real, energy weights remain binary64.
- MixtureThermo and chemistry numerical APIs: scratch Real independent of TableReal, default where needed preserves existing callers; table pointers never reinterpret-cast.

## Review focus and preregistered acceptance

1. Sub-ULP signed increments: stock 1e9 J, physical increments ±.02/.03/.05 J and -1e-4 J; actual final float flux versus independent decimal reference separately. Per-step storage error <=2 ULP(double stock), physical increment relative error <=0.2% for >=.02 J. Long sequence bounded by sum-of-roundoff, not a 73 J stock allowance.
2. Identity EOS and RK: valid stock bits unchanged by EOS and by zero-flux RK1/2/3, including 1/3 and 2/3; invalid legacy state still repaired. Mixture negative formation energy and high kinetic energy/small internal energy keep transactional output behavior.
3. ALE: nontrivial old/new volumes, exact double inventory formula using stored volumes, explicit chemical half-stage volumes; invalid geometry preserves all state, ALE RK2/3 rejected.
4. Lifecycle: typed binder, POD pointers, save/restore/retry, old float promotion, double restart round-trip; all-double owner tests unchanged. CHT native CUDA 32/64 translation units compile.
5. Sources: gravity, drag, volume fraction and heat read and retain small current-energy offsets; compare paired exchange against the actual published particle/source values and state remaining quantization limits.

## Task sequence

### 1. Shared storage, actual divergence and RK
- [ ] Add production-kernel regression fixture with independent double energy storage; show old code loses .02/.03/.05 J, fails typed binding, RK and EOS identity.
- [ ] Implement GasStateView/manifest/binder, energy arithmetic helpers and precise divergence/RK coefficient path.
- [ ] Run new tests in Real32/64 and existing gas transport probes; commit locally after evidence.

### 2. Recovery and chemistry
- [ ] Add mixed-table EOS/reactor tests, including Ns2/Ns10, negative formation energy, high kinetic energy, zero chemistry, source-free state identity and failure nonpublication; record RED.
- [ ] Generalize numerical scratch/table types; update energy adapter and preserve original control values and species precision.
- [ ] Run chemistry + mixture/ALE suites; compile CUDA fixtures; commit locally.

### 3. Production CHT owner and reachable sources
- [ ] Add real-source numerical tests retaining sub-ULP current-energy offsets, including actual gas/particle energy pairing; record RED. The latter currently fails conservation and blocks release.
- [ ] Promote only two CHT energy arrays; migrate gravity/drag/volume-fraction/heat energy arithmetic and audit every rhoE writer.
- [ ] Native compile CHT32/64, CHMT and gasUGKP affected fixtures. Record ptxas registers/stack/spills against baseline.

### 4. Restart, transactions and delivery verification
- [ ] Execute exact serialization/binder/mirror lifecycle checks and existing checkpoint/reject/retry suites. Register unsupported runtime checks honestly.
- [ ] Run full pytest collection with provided isolated dependencies; separate unavailable CUDA-runtime skips from failures.
- [ ] Record memory: CHT Real32 adds 4N bytes to each of rhoE/rhoENext = 8N total; no new double shadow or full-grid loop; Real64 owners add zero. Record any further allocation if introduced. CUDA events/memcheck require a GPU unavailable here.
- [ ] Independent review, focused local commits, no push/PR/merge from this worker. Parent owns integration and publication.

## Execution status and production release blocker

The shared storage, energy arithmetic, mixed thermo/chemistry, CHT owner and all reachable writer migrations are implemented locally with production-kernel host tests. Independent narrow fixes are committed separately: global dependency include order; lossless restart output; valid FP64 EOS energy and momentum preservation; exact zero-change FP64 RK; zero-time drag identity.

The actual-particle publication audit found a **new mixed-precision total-energy deficit** for sub-ULP particle heat updates: gas now retains the proposed heat decrement, while actual float pT and material-enthalpy moments stay unchanged. This is a release blocker, not an accepted conservation tolerance or merely an unchanged old limitation. Baseline local-kernel finite-step heat/drag residuals are reported separately; complete-advance conservation is not established by that fixture. Main precision production changes remain an uncommitted working patch while a paired actual-publication design is assessed. No rejection/trap, compensating heat, whole-owner FP64 promotion, or exchange-model change is introduced to hide this result. Any required scheduling change or persistent particle-energy state requires a separate design and user decision.

Native CHT32/64 and six mixed chemistry instantiations compiled on an earlier implementation snapshot. Final EOS/zero-exchange refinements are host-verified but have not been recompiled in CUDA. CHMT O2 was terminated with exit 137; its cause is not established. Per parent direction no additional heavy native build is launched here. No GPU runtime, memcheck, or CUDA event measurement is available.
