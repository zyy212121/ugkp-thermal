# CHMT shared reacting-wall adapter decisions

This is implementation and verification scope, not a physical validation claim.

## Selection and physical ownership

- `sst.wallTreatment` accepts existing `lowRe` and new `boundaryLayer`. Existing CHMT ordinary wallFunction coupling remains explicitly unsupported.
- `sst.boundaryLayer.model` accepts `reactingSst`, its `finiteRate` alias (the same solver with optional SST), and the restricted `constantTransport` frozen, constant-Cp, non-SST analytic limit.
- The finite-rate auxiliary profile owns no gas inventory. Only the two existing common chemistry half-steps update bulk gas chemistry.
- A physical material-wall species rate is the sum of material primary and pore rates after converting reference area to gas-face area. The matching-plane species flux and auxiliary reaction integral are never injected into the material ledger or coarse gas source.
- Packet energy contains physical species enthalpy with formation offsets plus absolute trace kinetic energy once, the shared wall conductive heat flux once, pressure/swept-volume work once and absolute viscous work once. Primary, pore Darcy and pore swept channels retain separate donor accounting.
- New-family gas contact resistance must be zero. Existing wet porous transfer, unsupported wet surface chemistry, resolved-normal film coupling and moving offset-surface restrictions remain in force.

## CPU and gas stage agreement

`evaluateInterfaceGasClosure` adapts the same common host/device core for CPU material temperature/composition iteration. Returned wall composition drives the next material constitutive evaluation. Film birth uses the shared incoming gas heat rather than the old linear conductance.

The material driver owns a canonical host mechanism copy for the enabled family. Mechanism and geometric views are transient and do not enter PhysicsConfig or checkpoints. CPU reacting profile scratch is one reusable heap workspace, shared serially across faces and temperature trials. Constant transport calls the analytic evaluator directly and allocates no profile scratch.

The gas stage supplies current material Tw, absolute wall velocity and physical species fluxes to the common sparse CUDA evaluator. Its matching primitive/SST state is read from current device arrays, using the certified positive donor stencil. Prepared outputs are downloaded once and split into CHMT primary/pore packets without a second profile solve. The shared preflight scheduler may reuse the first-stage profile only for the same generation, time and interval. A stationary profile failure is terminal for that coupling attempt in both CPU and gas adapters: halving a time step cannot repair an identical stationary BVP. Core status and nonlinear iteration are retained in the failure diagnostic.

For the existing supported film model, evaluation pressure and film drive pressure use the same stored matching pressure. The material lag controller also compares matching-state changes, so an unchanged owner cell cannot hide a changing wall pressure. Intermediate material endpoints follow the existing representative-stage history convention; the macro endpoint instead samples the actual endpoint gas inventory on its certified endpoint geometry. One sparse endpoint matching sample is cached per material candidate when the new wall family and film are enabled. Its pressure drives the film endpoint pV conversion; no last-stage pressure is relabelled as endpoint data. No film equation or support range is changed.

A driving history record stores one coherent representative gas stage: owner primitive, gradient, traction, matching state and explicit `gasTraceTime`. The exchange packets remain separately RK-weighted time integrals. This fixes the former endpoint-primitive/last-substage-gradient mixture; material interpolation still uses the existing piecewise-held history convention, now with all fields at the same representative time.

## Geometry and cost boundaries

The adapter delegates certification, actual-volume quadrature and connected matching-ray construction to the common geometry builder. It marks all physical no-slip/interface walls, but selects only the coupled material surface for the new closure. Other walls keep their existing low-Re treatment.

Reacting SST owner-source quadrature is cut at the configured profile nodes and uses the common first-segment order-8 positive Gauss rule; scalar normal weights are aggregated before device upload. Nodes/stretch and the effective SST selector determine this geometry preprocessing. Both host and device-adapter outer caches include every effective geometry option, including the fixed Gauss order, and retain unchanged descriptors without a new mesh traversal. Existing nodes/stretch/SST restart identity already covers all user-controlled inputs, so no schema change is needed. Descriptors and device geometric arrays are rebuilt only when geometry or effective quadrature options change. Moving rollback/alternate candidates explicitly invalidate the cache because a rejected branch can reuse a geometry version for a different shape. Static rollback does not rebuild geometric descriptors.

The enabled family adds one sparse profile kernel per distinct gas stage, plus sparse status/output/matching-input device-to-host copies needed by the existing host-owned material packet interface. These copies synchronize the stage. They are not native performance measurements. No profile workspace, wall arrays, new profile kernel or new matching history is allocated for an unchanged low-Re configuration.

GPU reacting profile scratch uses a bounded reusable worker pool rather than one large workspace per face or device-thread automatic storage. Nodes range from 4 through 128, with capacity buckets 24, 48 and 128; the default remains 24 and resolution must be refined for the particular case. `workspaceSlots 0` is automatic: desired workers are 32 times the actual device multiprocessor count; scratch budget is min(free device memory / 8, 1 GiB); resolved slots are min(face count, desired workers, budget / workspace bytes, 4096). A positive explicit slot count is capped by faces and must fit available memory. Allocation happens after gas state and wall geometry allocation. The backend reports requested/resolved slots, capacity, bytes and budget. This is an initial bounded policy, not a measured throughput optimum. Constant transport allocates no scratch and launches one thread per selected face. CPU remains serial with one reacting workspace; host CUDA shims use one deterministic worker without querying or claiming GPU properties.

## Diagnostics and restart

The last accepted profile diagnostic is copied with accepted HostState, so rejected trials and failed coupling candidates do not publish it. `wall-layer-<accepted step>.csv` records the actual representative stage time, physical versus matching species flux, auxiliary reaction integral, species balance residual, wall heat/shear and SST outputs.

An unavailable coarse/profile chemistry comparison is explicitly flagged and its CSV field is empty. It is never emitted as a finite zero or NaN numeric value. The diagnostic does not enter any mass, energy, reaction or material audit.

Profile diagnostic snapshots are reconstructible outputs and are intentionally not checkpointed. A restart first reports `NOT_AVAILABLE`; the next accepted gas stage supplies a new diagnostic. Authoritative wall-family controls are checkpointed in schema 6, including solver and workspace controls, so changing them rejects an incompatible restart. Physical matching histories are retained during the provisional coupling window, independently of reconstructible output snapshots.

## Evidence boundaries

Focused executable tests cover physical/auxiliary flux separation, candidate-temperature and blowing-velocity consistency, analytic shear, exactly-once energy terms, film-birth heat, canonical finite-rate chemistry against an independent reaction-diffusion solution and frozen control, prepared-output reuse, matching-state freshness, bounded scratch storage, stage coherence for Euler/RK2/RK3, output availability and restart identity.

Host CUDA shims execute actual shared functions sequentially; they are neither native CUDA execution nor a performance result. Native compiler/link results and actual OpenFOAM CPU runs are reported separately. Physical wall-resolved convergence, 3-D turbulent reacting/blowing benchmarks and engineering GPU cases are separate gates.

The additional actual OpenFOAM film regression prescribes 100 kPa owner pressure with 150/180 kPa matching-stage pressures and a distinct 210 kPa endpoint matching pressure (90 kPa endpoint owner). It requires two material slabs, correct film endpoint pressure/temperature, a 130 J pV change and a closed gas-plus-material-plus-film internal-energy ledger. The supplied gas endpoint change is explicitly accounted by its external boundary budget; this is a pressure-adapter/ledger regression, not a gas CFD validation.
