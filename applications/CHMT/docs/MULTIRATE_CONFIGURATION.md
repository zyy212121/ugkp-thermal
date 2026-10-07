# CHMT CPU-material / GPU-gas execution

The coupled material frontend selects `executionMode Multirate` by default.
The actual OpenFOAM solid-region `fvMesh` remains alive for the CPU implicit
material solve. The CUDA backend advances gas, SST and supported particles in
its own microsteps. CPU material, reactions, Darcy transport and surface-film
updates run only in the material candidate, on independently selected substeps.

This source implementation still requires real Foundation OpenFOAM 10 and CUDA
build/runtime verification. Host unit checks are not a substitute for either.

## Independent clocks

Add the following to `constant/chmtProperties` for a coupled case:

```
executionMode Multirate;
multirate
{
    couplingInterval          1e-3;
    minimumCouplingInterval   1e-9;
    gasMaxDt                  1e-5;
    materialMaxSubstep        0;
    maxWindowIterations       8;
    maxWindowRetries          12;
    enforcePredictionAccuracy false;
    predictionTolerance       1;
}
```

These numbers illustrate the configuration syntax; they are not validated
physical settings for any material or mesh.

- `couplingInterval` is required and selects the desired macro exchange interval
  H. Output/end-time boundaries and failed/event-limited windows can shorten it.
- `gasMaxDt` caps gas microsteps. It defaults to, and cannot exceed,
  `controlDict.deltaT`. The backend also applies gas-specific restrictions.
- `materialMaxSubstep` is a CPU accuracy cap, independent of gas CFL. Zero means
  no additional user cap; the material driver still applies its internal
  admissibility, transport, reaction, nonlinear and event restrictions.
- `minimumCouplingInterval` defaults to H times 1e-6 and bounds macro retries.
  A shorter final output landing can be accepted without inventing a gas step.
- `maxWindowIterations` bounds waveform replays for one interval length.
  Exhaustion may shorten the interval within `maxWindowRetries`.

Optional `cpuMaterial` entries are `transportCfl`, `reactionCfl`,
`reactionMaxSubstep`, `maxSubsteps`, `nonlinearMaxIterations`,
`nonlinearRelativeTolerance`, `energyAbsoluteTolerance` and
`energyRelativeTolerance`. They never enter the gas CFL bound.

Three additional CPU driving controls govern bounded, first-order coalescing of
recorded gas driving states: `driveRelativeTolerance` (default 0.05, nonnegative),
`driveSourceFraction` (default 0.05, greater than zero and at most one), and
`driveTractionScale` (default 1 Pa, positive). Zero relative tolerance requests no
approximate coalescing. Exact extensive transfer integrals are retained.
These are local lag/substep controls, not a demonstrated global error bound;
accuracy requires independent refinement. Reports distinguish boundary-drive
sampling from full material RHS evaluations.

## Coupling semantics

The default is a declared first-order lagged window. A changed wall temperature
or constitutive mass-rate proposal may be used at the next synchronization
without replaying a conservative/admissible current window. This is not a claim
of within-window constitutive convergence or second-order temporal accuracy.
Independent H refinement is required to establish coupling accuracy.

Every accepted window still requires:

1. Actual accepted gas transfer integrals consumed exactly once by the CPU
   material/film owner, with matching mass, species and energy accounting
2. Conserved total mass/elements and physical energy under the model's explicit
   boundary/work/radiation and reduced-film kinetic-defect accounts
3. Valid shared donor histories and unchanged phase/radiation ownership
4. Material geometry matching the actual executed gas trajectory

Geometry or phase/regime disagreement requires replay or an event-aligned
shorter window. Setting `enforcePredictionAccuracy true` additionally requests
waveform iterations until the normalized temperature/mass prediction error is
at most `predictionTolerance`; normalization uses the existing model tolerances.

## Output and restart

Only synchronized macro commits are output/checkpoint authority. A failed window
leaves the last accepted CPU and GPU inventories unchanged; the frontend attempts
a `last-accepted` synchronized checkpoint. There is no pending-window resume.
Restart rebuilds a predictor from the synchronized checkpoint and the supplied
case controls. `HostState.nextDt` represents only the next gas microstep cap in
multirate mode; H comes independently from the coupling dictionary.

- `multirate.json` records effective clocks, lagged/iterated mode and checkpoint
  policy.
- `windows.csv` records H, next gas/macro intervals, executed and accepted gas
  microsteps, CPU solve/substep counts, boundary-drive samples, material RHS
  evaluations, retries and prediction/audit residuals.
- `raw-solid-<step>.csv` contains accepted extensive material inventories and
  recovered temperature/porosity/pore pressure. The full checkpoint retains
  gas, solid, film, geometry, particles and budgets.
- With `writeStageGeometry true`, geometry from every accepted microstep of the
  committed replay is retained. Face files and dimension-independent `gcl.csv`
  rows use the actually executed sweeps. Rejected epoch geometry is discarded.

## Explicit separate routes

`executionMode LegacyExplicit` retains the original all-explicit CUDA evolution
as a regression/reference route. `executionMode Standalone` is limited to models
with no coupled solid/interface exchange. Existing explicit standalone-surface,
normal-column, and gas-only cases select Standalone automatically when no mode
is supplied. Local profile/pressure/phase and remap operators do not require a
multirate dictionary. No new normal-column production model is introduced.

Finite-duration or already-pending particle contact is not supported by the new
window path. Existing unsupported moving wet-offset geometry, topology changes,
nonconformal coupling and material-model restrictions remain explicit errors.
A shorter interval is not permission to silently approximate an excluded model.

## Build separation and checks

`bash applications/CHMT/Allwmake` builds the CUDA backend separately, then uses
host `wmake libso` for `libCHMTCpuMaterial` and links the frontend against both.
OpenFOAM headers and CPU matrix sources are not sent to nvcc.

`bash applications/CHMT/devtools/check_multirate_frontend.sh` checks source/build
routing and host transaction control with scripted participant responses. It
checks scheduling, replay, rollback and audit gating; it executes no CFD and
makes no OpenFOAM/CUDA numerical validation claim.
