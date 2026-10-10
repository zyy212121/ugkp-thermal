# Coupling-only CHMT common-backend rebuild

CHMT is a coupled gas/material application. `executionMode Multirate`, populated
gas and material regions, and a mapped interface are mandatory. There is no
Standalone entry or independently restored gas solver. Pure-flow verification
belongs to gasUGKP.

## Build and verification boundaries

`Allwmake` now builds the native CUDA backend plus the real OpenFOAM 10 material
coordinator. Root `Allwmake` includes CHMT. Set `CHMT_CUDA_ARCH=sm_XX` for the actual
GPU and `UGKWP_GAS_SPECIES` for the exact material species metadata count.
`CHMT -build-info` prints source/compiler/build identity; native acceptance also
records executable SHA-256 and the build manifest.

`Allwmake --frontend-only` produces `CHMT-input-check`. It imports real coupled
input and rejects evolution. Its build identity explicitly says COMPILE_ONLY.
`devtools/check_runtime_frontend.sh` parses actual runtime C++ branches against
OpenFOAM; it does not compile CUDA.

Native CUDA is unavailable in the current environment. The native thermal,
diffusion and reacting-gas acceptance commands are implemented but NOT_RUN.
The exact commands, independent conservation/caloric/thermal-magnitude checks,
control runs, refinement and checkpoint/restart comparisons are in
`verification/coupled_runtime/ACCEPTANCE.md` (including `--single`). Passing host or import checks must
never be represented as native GPU execution or parallel correctness.

`devtools/check_host_coupled_pipeline.sh` is an explicitly labelled supplemental
check: it links the actual OF10 material coordinator and executes the common
kernel bodies sequentially on the host. It is useful for whole-call-path defects
and independent output balances, but is not a substitute for the native gate.

## Public numerical ownership

`gpu/Backend.cu` owns storage, coupled wall flux replacement, trial snapshots,
donor reservations, histories and publication. It includes the actual common
flux, gradient, limiter, turbulence and positivity operators and invokes
`common/GpuGasAdvance.cuh` for Euler/SSPRK and chemical source composition. It does
not contain a second CHMT gas flux/reconstruction/time-integration implementation.

The shared gas model and mechanism parsers own ordered species, canonical thermo,
optional diffusion, optional homogeneous chemistry and chemical controls.
Gas/pore calorics evaluate common MixtureThermo. Material/surface reactions and
condensed/film thermodynamics retain their distinct physical owners.

Supported combinations are checked through the common capability gate. Mixtures
use Rusanov or Kurganov/HLL, first-order or composition-aware MUSCL, common
limiting, and static Euler/SSPRK2/SSPRK3. Moving geometry uses certified Euler
old/new volumes and face sweeps from the shared ALE policies. Both legacy and
mixture low-Re SST use common source/transport/projection audits; those are
integral diagnostics, with net source distinct from gross production/dissipation.
High-Re coupled SST, particles, resolved-normal coupled film, and nonzero gas
gravity are explicitly rejected where their matching coupled implementation is
not supplied. They are compatibility limitations, not silently disabled physics.

### Explicit supported capability combinations

| Configuration | Gas transport owner | Geometry and time stepping | Coupled restrictions |
| --- | --- | --- | --- |
| Single legacy, laminar | Common legacy flux/reconstruction/limiter | Static Euler/SSPRK2/SSPRK3; certified Euler ALE | One active constant-Cp, zero-formation species; no gas species arrays |
| Frozen mixture, laminar | Common Rusanov or Kurganov/HLL; first-order or MUSCL; common limiter | Static Euler/SSPRK2/SSPRK3; certified Euler ALE | Exact compiled metadata count/order; molecular diffusion optional |
| Reacting mixture, laminar | Same common transport plus shared symmetric chemical half steps | Same static/ALE limits; explicit old/new chemical volumes | Immutable configured mechanism; formation-inclusive total energy |
| Low-Re SST, single or mixture | Same common SST operators and integral source/constraint audit | Static Euler/SSPRK2/SSPRK3; certified Euler ALE | Low-Re treatment only; configured coefficients and boundary values retained |
| Thin averaged film/material | Restored CPU material/film coupled to the above gas owner | Coupled recession geometry and macro replay | Native wet/receding coupled GPU acceptance still required |
| Particles, resolved-normal coupled film, nonzero gas gravity, high-Re coupled SST | No silent fallback | Reject explicitly | Not claimed restored by this rebuild |

All supported rows remain subject to native CUDA acceptance. ALE
energyLimitedLinear and mixture fluxes without conservative species splitting are
rejected by the shared capability gate.

## Optional single-component mode in the same binary

For `constant/gasModelProperties` with `gasMode single`, supply the complete real
material/pore species metadata in `constant/materialGasProperties`, using the
same canonical parser and `gasMode mixtureFrozen`. Select one actual metadata
name with `singleGasSpecies` in `constant/chmtProperties`. This uses the same
compiled executable as the mixture path and allocates no device gas species
arrays. All material metadata entries must be physical species; no dummy padding.

The active species must have constant Cp, positive Cv, and zero formation-energy
offset to agree with the unchanged legacy five-field sensible-energy EOS.
Initial gas, pore gas, film inventory, boundary inflow, material/surface reaction
emission and wall programs cannot introduce another species. Variable-Cp/NASA7
single-species physics can instead use the general mixture mode at its exact
species count. Molecular multicomponent diffusion is disabled in legacy mode.

## Transactions and persistence

Accepted macro state is immutable while gas microsteps and CPU candidates run.
Rejected microsteps restore state, geometry, SST/chemical audits and budgets.
A failed device rollback is reported and makes that backend unusable; no retry
can publish a partially restored device state. Reservation-rejection rollback
failures stop immediately as nonrecoverable, including inside microstep retry loops.
Gross gas donor checks keep primary, Darcy and geometric sweep withdrawals distinct,
aggregate all interface faces per cell/species, and validate combined outward
transport after the unchanged common positivity limiter. Incoming counterflow cannot
finance an explicit gas withdrawal. Material commits cannot alter the
accepted gas/SST endpoint. Donor/history bounds and exact endpoint clocks are
checked before synchronization and publication.

Schema 5 checkpoints contain immutable model/species/thermo/mechanism identity,
the single active species index and accepted cumulative SST/chemical diagnostics.
Borrowed pointers are not serialized. Older schemas reject. Initial and accepted
macro outputs include gas/material inventories, interface transfers and synchronized
checkpoint state for independent checks.

The restored interval audit was corrected: zero-sum exchange residuals use the
magnitude of participating transfers for relative normalization, while preserving
absolute tolerances and independent recipient checks. This prevents cumulative
pressure-impulse subtraction roundoff from causing false macro rejection; genuine
owner misattribution and mismatches still reject. The restoration audit classifies
this as an adapted source, not unchanged historical mathematics.

## Selected restored material components

Material, film, interface, geometry, donor/ledger and restart pieces originate
from `e9bc511a4f1feffec9ef5fc17024bfa40a9b4610`. Run
`python3 devtools/restoration_audit.py --verify-reference` to distinguish unchanged
sources, adapted interfaces and new code. Historical hashes remain immutable in
`restoration-source-manifest.json`; `rebuild-adaptation-manifest.json` records the
current reviewed source changes, including the interval audit and new backend. No historical CHMT gas flux, gas kernels,
SST kernels, independent backend or pure-flow benchmark was restored.

Useful gates:

- `bash devtools/check_host.sh`
- `bash devtools/check_frontend.sh` (same-binary single and mixture input)
- `bash devtools/check_chemistry_frontend.sh` (exact Ns=10 mechanism import)
- `bash devtools/multirate/check_cpu_material_of.sh` (actual OF10 sparse solves)
- `bash devtools/check_host_coupled_pipeline.sh [--chemistry]` (supplemental host operators, controls and restart)
- `bash devtools/check_native_coupled.sh /tmp/new-evidence [--diffusion|--chemistry]`

No README files are part of this rebuild.
