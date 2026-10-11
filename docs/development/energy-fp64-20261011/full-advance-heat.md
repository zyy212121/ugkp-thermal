> Historical evidence note (updated 2026-10-11): blocked-migration and old
> coupling conclusions below describe the pinned earlier snapshot, not the current
> paired-carry implementation. The current design/cost contract is
> [paired-carry-design.md](paired-carry-design.md); current verification and
> remaining native-runtime limits are in [current-verification.md](current-verification.md).

# Heat source, recovery, and particle cache: executable ordering evidence

## Result and scope

`tests/gas_transport/test_full_heat_advance.py` now contains two deliberately
separate kinds of evidence:

1. A completed HOST execution of the actual source/cache/recovery/particle
   numerical bodies, with the production alumina caloric law. It verifies the
   surrounding order and full-dt call arguments against the actual strict
   Advance source and its callers. It **does not execute full strict Advance**.
2. An opt-in native CUDA fixture which includes `GpuResidentStrict.cu`, uses its
   actual allocator and moment initialization, and calls its entire exported
   `ugkwpGpuResidentStrictAdvance`. Its generator checks pass. **It has not been
   compiled or run at this historical audit checkpoint** because that audit
   did not have a native CUDA compiler. Later integrated native compilation is
   tracked separately; this old availability statement is not a current
   environment claim. A generated fixture is not a successful CUDA build or a
   full-solver conservation result, and actual GPU execution remains NOT_RUN.

No production source, heat law, scheduling, drag law, momentum precision, or
README was changed for this evidence. This is characterization before the
separate particle thermal-carry implementation; it is not acceptance of that
implementation or permission to replace the heat law.

## Executed source path

The HOST fixture reuses existing storage/CUDA-syntax adapters, but replaces all
three earlier linear material adapters with these production headers:

- `common/operators/particleSpecificHeatDevice.cuh`
- `common/operators/particleTemperatureFromSpecificEnthalpyDevice.cuh`
- `common/wall/GpuAluminaProperties.H`, reached through actual wall headers

It executes, in order:

1. Actual atomic particle moment accumulation/normalization and actual solid
   recovery to construct the initial material state.
2. `applyGasVolumeFractionSourceKernel` and `recoverPrimitivesKernel`.
3. `snapshotParticleGasCouplingStateKernel`, in the heat-enabled/drag-disabled
   branch used by strict Advance.
4. `applyEulerianParticleMaterialHeatKernel` with full dt.
5. `recoverPrimitivesKernel`, including actual gas primitive recovery, then the
   additional recovery which strict Advance runs before the collision pool.
6. `relaxOneParticleToResidentGas` with full dt.
7. Actual particle moment reaccumulation, solid recovery, and a final actual
   gas/solid primitive recovery.

The extra final HOST recovery checks whether recovery could close the observed
gap. It is an explicitly identified extra check, not a claim that this call
appears at that precise location in strict Advance.

Disabled drag adapters abort if reached. Unsupported wall side-effect adapters
also abort if reached. No fake numeric collision, pressure, tracking, flux, or
compaction function is substituted and called a full solver.

The source-order test reads the actual `ugkwpGpuResidentStrictAdvance` body.
It verifies flux, volume source, coupling snapshot, heat source, injection,
directory preparation, pre-pressure, collision correction, cold-wall/mobile
relaxation, tracking, moments, post-pressure, and boundary-finalization order.
The application calls `advanceOneStep(deltaTValue(), value())`; that wrapper and
both precision-dispatch branches forward dt unchanged. Heat and ordinary
particle relaxation use full dt. The two collisional-pressure calls alone use
`0.5*dt`. The flux RK stages complete earlier; they do not wrap the heat calls.

## Actual cache observation

For finite heat, production gas recovery lowers current `Tgas` from 3200 K to
approximately 3189.4749 K. It leaves `couplingTgasOld` at 3200 K. The actual
particle publication equals the full-dt response to the saved coupling cache.
It differs from both:

- The same production response evaluated with dt/2.
- An actual counterfactual particle relaxation with the saved cache explicitly
  replaced by recovered current `Tgas`.

Thus gas recovery does not silently make the independent particle heat update
use the new gas temperature. This is now executed evidence, not solely a text
search or a reconstructed intended transfer.

## HOST configuration and all excluded stages

The HOST compartment has volume 1 m³, gas density 1 kg/m³, one represented
100 kg mobile parcel, solid density 1000 kg/m³, diameter stored as Real(0.001),
particle temperature 2800 K, and gas temperature 3200 K. The production
piecewise nonlinear alumina law is active at these temperatures.

| Stage | HOST status | Native CUDA fixture status/required check |
|---|---|---|
| Gas flux, diffusion, boundary transport | Unsupported, not executed; the compartment has no mesh faces | Actual Euler flux pipeline on a six-face stationary cube; every boundary mass and energy flux must be zero |
| Gas/particle gravity | Explicitly zero; drag disabled, so particle force integration is not entered | `gravityEnabled=0`, zero gravity |
| Gas pressure force in particle drag | Disabled by `dragModelId=0`; unavailable adapters abort | Actual disabled branch |
| Volume-fraction source | Actual execution; no spatial gradient and unchanged `epsGPrev`; its publication is checked unchanged | Actual execution on a closed cube with constant volume fraction |
| Injection | Not executed; unsupported in HOST | `nBoundarySources=0` |
| Pre-directory preparation | Not executed; one explicitly identified particle | Actual full/split CSR protocol |
| Both collisional-pressure half-steps | Not executed; no HOST claim | `collisionalPressureEnabled=0`, actual early-return branch |
| Collision pool and correction | Not executed; no HOST claim | Actual kernels; zero granular temperature, zero resulting pool mass required |
| Cold-wall 1D/2D, finite contact, wall heating | Mobile parcel, no wall contact; unsupported side effects abort | Both cold-wall models and wall heating disabled; parcel must remain mobile |
| Packing projection | Not executed | `jammingPressureEnabled=0`, actual disabled branch |
| Tracking | Not executed; no HOST mesh/CFL claim | Actual tracking of a motionless centre parcel; position, cell, status and identity must remain unchanged |
| Post-transport moments/recovery | Actual atomic moments and solid recovery, but no preceding transport | Actual resident CSR moment pipeline |
| Compaction | Not executed | Actual resident gather/commit; count and original ID checked |
| Boundary finalization | Not executed | Actual resident finalization |
| External CHT/radiation after Advance | Unsupported and absent | Absent from this isolated exported-Advance driver; no external coupler invoked |

The finite HOST dt=0.001 s is a local source/cache stress point. No production
mesh CFL is asserted for it. The CUDA driver instead uses dt=1e-4 s for finite
heat and calls the actual gas Courant API, requiring returned maxCo < 0.5. Its
microheat dt is 1e-8 s. These native preconditions are assertions awaiting a real
run, not presently verified facts about execution.

## Published-state ledger

The main independent budget is

`epsG*rhoE + m*(u²/2 + 1.5*theta + h(Tstored))`.

Python Decimal evaluates the production piecewise caloric definition at the
actual stored particle temperature, using the actual precision-specific
material coefficients. It does not use intended dHp or an intended temperature
as particle inventory. Raw rounded h(T), material/mechanical cell moments, gas
conserved fields, and both temperature caches are also captured. Differences in
rounded moment publication are not silently substituted for physical particle
energy. Each inventory is evaluated at 70-digit precision; printed differences
are rounded to the Decimal context used for reporting.

Results before thermal carry, measured 2026-10-11:

| Real bits | Case | Exact old total ΔE (J) | Current total ΔE (J) |
|---|---|---:|---:|
| 32 | Heat disabled | 0 | 0 |
| 32 | Microheat, dt=1e-8 | 0 | -0.06887998398462969 |
| 32 | Finite heat, dt=0.001 | +105.11116861580147 | +105.09907450549886 |
| 64 | Heat disabled | 0 | 0 |
| 64 | Microheat, dt=1e-8 | +2.170649393744142e-9 | +2.170649393744142e-9 |
| 64 | Finite heat, dt=0.001 | +91.02531206511845 | +91.02531206511845 |

Current zero-time cases also preserve all checked energy/particle publications.

In mixed32 microheat, actual pT stays exactly 2800 K. The current gas loses
0.06887998398462969 J; its energy remains identical through all subsequent
executed recovery and moment stages. The exact old gas and particle discard
this transfer. The old microheat coupling cache is 3199.999755859375 K versus
current 3200 K, a consequence of recovering old FP32 gas energy; both actual
particle publications remain 2800 K.

For finite heat the independent residual is already present in exact old source
and in Real64. This is a source/cache composition observation; it is **not**
labeled a demonstrated full-production-step defect or a migration-caused drag
defect. No unexecuted-stage cancellation has been assumed either way.

The optional baseline used here is the same `ab900f54` extraction used for
`particle-budget.md`, with its documented include-order compatibility fix.
The baseline restores homogeneous original gas-energy storage rather than
running old kernels through new double gas pointers. Particle relaxation and
alumina header hashes are identical between the compared sources:

- `GpuThermalParticleRelaxation.cuh`: `690a17b609ba8d3e1d86aff70fe5c4b5ace0676a18f99dc787129f67a7af59b3`
- `GpuAluminaProperties.H`: `160be19a69a06bd7277d99d9f36fad83c0bbaae3449c364560ccbde4866f4ce5`

## Running and verification limits

Focused command (pytest must be available):

```sh
UGKWP_PARTICLE_BUDGET_BASELINE_ROOT=/path/to/exact/ab900f54 \
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/gas_transport/test_full_heat_advance.py -q -s
```

Observed: **14 passed, 2 skipped**. The skips are the deliberately opt-in native
CUDA builds. Without a baseline root the two baseline comparisons also skip.
The combined run with `test_actual_particle_energy_budget.py` and
`test_authoritative_energy_sources.py` completed with **49 passed, 2 skipped**.
Tests were written before filling in the new HOST fixture; its absent production
header coverage assertion was observed failing, then passed after implementation.

Native compile only, on a CUDA-enabled machine:

```sh
UGKWP_FULL_HEAT_CUDA=compile UGKWP_CUDA_ARCH=sm_89 \
python -m pytest -p no:cacheprovider \
  tests/gas_transport/test_full_heat_advance.py -k opt_in_native -q
```

Use `UGKWP_FULL_HEAT_CUDA=run` to build and execute instead. The architecture and
CUDA_HOME must match the target. Each executable includes the real resident
source and links `GpuWallEnergy64.cu`; it prints actual before/after fields and
the independently calculated budget. Native compilation/runtime remain
unverified here, and any build/runtime failure must be reported before using
the fixture as full-Advance evidence.

A root `python -m pytest -p no:cacheprovider -q` was attempted. It stopped during
collection with 38 existing cross-application import/configuration errors; no
repository-wide pass is claimed. Three CHT devtools modules cannot resolve
`devtools` under that root invocation. The other listed modules collide with
same-basename modules in other application trees. No unrelated cleanup was made.

Affected collection modules, grouped by directory:

- `applications/CHT/devtools/tests/`: `test_acceptance_gates.py`,
  `test_generate_cht_perf_particles.py`, `test_long_run_audit.py`
- `applications/FSH/devtools/tests/`: `test_courant_update_contract.py`,
  `test_sst_algebra.py`, `test_wale_compressible_invariant_contract.py`
- `applications/FSH/tests/`: `test_collision_sampling_contract.py`,
  `test_gas_solid_coupling_contract.py`, `test_gpu_boundary_schedule_contract.py`,
  `test_jamming_pressure_contract.py`, `test_restart_tools.py`,
  `test_upgrade_contract.py`
- `applications/gasUGKP/devtools/tests/`: `test_courant_update_contract.py`,
  `test_sst_algebra.py`, `test_turbulence_port_contract.py`,
  `test_wale_compressible_invariant_contract.py`
- `applications/gasUGKP/tests/`: `test_boundary_ce_gradient_contract.py`,
  `test_boundary_eos_contract.py`, `test_collision_sampling_contract.py`,
  `test_epsg_prev_restoration.py`, `test_full_gks_math.py`,
  `test_gas_solid_coupling_contract.py`, `test_gks_les_contract.py`,
  `test_gpu_boundary_schedule_contract.py`, `test_identity_contract.py`,
  `test_interface_contract.py`, `test_jamming_pressure_contract.py`,
  `test_no_wall_retention_contract.py`, `test_numerics_config.py`,
  `test_openfoam_boundary_configuration_contract.py`,
  `test_openfoam_case_configuration_contract.py`,
  `test_openfoam_energy_interpolation_contract.py`,
  `test_openfoam_viscous_flux_contract.py`,
  `test_particle_cell_path_runtime_contract.py`, `test_tau_wave_contract.py`,
  `test_wall_gks_contract.py`
- `examples/consistency/windSandShockTube/assets/solver_source/tests/`:
  `test_constant_response_time_contract.py`, `test_staged_coupling_contract.py`

## Consequence for the repair

The microheat result supports retaining particle material energy beyond FP32
temperature/profile publication. The approved specific-enthalpy carry must be
included in inventory as `m*(H(stored profile or T)+carry)` and used by future
thermal/phase updates. These pre-carry characterization tests do not establish
that lifecycle property.

Any later heat-pairing allocator must distinguish actual received material
energy from the old gas-side finite-capacity proposal, and must preserve shared
bounds. That proposed numerical contract remains unimplemented here. The finite
baseline residual alone does not authorize a new heat law or broad drag rewrite.
