# CHMT build, input and adapter contract

CHMT is a new, independently built Foundation OpenFOAM 10/CUDA application.
It does not modify the original applications, root build entry, common-source
mirror lists, or the separately published verification package.

## Evidence boundary

Host mathematical, serialization, dictionary and subprocess tests are software
checks. A frontend object compile is **COMPILE_ONLY**: it does not produce a
runnable solver or establish CUDA compilation, linking, kernel safety, numerical
accuracy, convergence, or coupled CFD validation. There is no CPU/mock solver
fallback. Actual CUDA/GPU and CFD validation must be performed separately.

The adapter lists dispatches present in the source. `UNVERIFIED` remains the
validation status even after successful execution. The three local film cases
are CUDA operator contracts, not time-evolved CFD tests. Remap is one real CUDA
piecewise-constant overlap operation, not general 3-D remeshing.

## Build and installed paths

Source an actual Foundation OpenFOAM 10 installation. Select a CUDA architecture
supported by the installed nvcc and an explicit compile-time ordered species set:

```sh
source /path/to/OpenFOAM-10/etc/bashrc
CHMT_CUDA_ARCH=sm_80 CHMT_SPECIES=S0,S1 bash applications/CHMT/Allwmake
```

`S0,S1` are the two dimensionless mathematical verification species only. They
are not a production mechanism. For real material work, provide the complete
ordered species set and matching material/kinetic tables. Changing species
requires rebuilding; dictionary order must match every compiled name exactly.

Use `bash` for source scripts: website-uploaded source files can have mode 0644.
Allwmake installs a mode-0755 adapter as a build output, without changing source
modes. Default outputs are:

- `applications/CHMT/.build/bin/CHMT`
- `applications/CHMT/.build/bin/chmt-adapter`
- `applications/CHMT/.build/bin/chmt-adapter.build.json`
- `applications/CHMT/.build/lib/libCHMTBackend.so`
- `applications/CHMT/.build/generated/{CHMTSpecies.H,CHMTBuildIdentity.H,build.json}`

Set `CHMT_BUILD_DIR` to a dedicated absolute directory outside the checkout if
needed. The only allowed in-tree build directory is CHMT/.build, so generated
identity never recursively enters source hashing.
Only CHMT-named artifacts are installed; the original solver binaries are not
replaced. `NVCC`, `CXX` and `CHMT_CUDA_LIBDIR` can select the real toolchain.
Missing nvcc, unsupported architecture, wrong OpenFOAM version, missing runtime
source or compiler failure exits nonzero. It never substitutes exported fake
Backend symbols or silently changes precision. Only FP64 is supported.

The generated identity contains the actual checkout HEAD, dirty-source status,
full CHMT source fingerprint, directly/transitively included common-source
hashes, true upstream base, compiler versions, CUDA architecture, OF version,
Ns and ordered species. The recorded upstream base is
`db455604156419a9e20b13f1b43694fde33be6c8`; it is distinct from the implementation
checkout revision. Any earlier synthetic preparation checkout is not claimed
as a remote commit. The user's own checkout revision and source bytes are
resolved anew when they build. Generated identity/output/cache files are excluded
from source hashing to avoid circular fingerprints.

A build-info-only query is `CHMT -build-info`. Hardware comes from the actual
CUDA device query; an unavailable device is explicitly reported and execution
cannot produce a successful run. Runtime configuration cannot replace the
compiled source identity with an arbitrary supplied string.

## Host checks, without a CFD run

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s applications/CHMT/tests -p test_adapter.py
PYTHONPYCACHEPREFIX=/tmp/chmt-python-cache python3 -m py_compile \
  applications/CHMT/adapter/chmt-adapter \
  applications/CHMT/devtools/build_info.py \
  applications/CHMT/devtools/source_fingerprint.py
bash applications/CHMT/devtools/check_frontend.sh
PYTHONDONTWRITEBYTECODE=1 python3 applications/CHMT/devtools/source_fingerprint.py
```

To include read-only external-spec generation and tiny actual mesh/initialization
checks, set CHMT_TEST_PACKAGE to the separate test/chmt directory and
CHMT_OPENFOAM_HOST_CHECKS=1 for the unittest command. Those host checks hand-author
tiny topology fixtures, import them through OF10, and check initial averages;
they never invoke blockMesh, CHMT evolution, or CUDA.

The object checker uses real OF10 headers and a deliberately non-runnable
COMPILE_ONLY identity. Its temporary files are removed. It does not link CUDA,
create CHMT, or execute blockMesh. The adapter tests use conspicuously labelled
subprocess fixtures for command/error/provenance behavior only. Fixture data
are never observations submitted to the numerical verification suite.

## Running the separately published unchanged test package

Keep the test PR/package in a separate checkout. No test path is hard-coded into
the adapter and no test files are copied into this implementation. For example:

```sh
TEST_PACKAGE=/absolute/path/to/separate/test-checkout/test/chmt
CHMT_BIN=/absolute/path/to/implementation/applications/CHMT/.build/bin
python3 "$TEST_PACKAGE/run_case.py" prepare --case film_profile \
  --output /absolute/path/to/fresh-film-profile
python3 "$TEST_PACKAGE/run_case.py" run --case film_profile \
  --output /absolute/path/to/fresh-film-profile \
  --adapter "$CHMT_BIN/chmt-adapter" --solver "$CHMT_BIN/CHMT"
```

The exact direct adapter interface is:

```sh
"$CHMT_BIN/chmt-adapter" --describe
"$CHMT_BIN/chmt-adapter" run --spec /absolute/prepared/case-spec.json \
  --solver "$CHMT_BIN/CHMT" --output /absolute/prepared
```

Supported mathematical dispatches: `film_profile`, `film_pressure`,
`film_phase_energy`, `film_plug`, `film_shear`, `ale_free`, `ale_wave`, `remap`,
and `stefan`, subject to the actual compiled/runtime capability checks. Freno
MMS and TACOT remain DATA_REQUIRED and are refused; no missing material fixture,
reference curve or tolerance is invented.

The adapter hashes and preserves the original spec, request and reference bytes.
It does not import or execute reference.py, or inspect reference values to
compute observations. It refuses existing run artifacts, missing executables,
unsupported schema/cases, failure return codes, incomplete raw output, wrong
build/species provenance and unavailable hardware. It requires typed CUDA
availability and a self-consistent compiled manifest; expected execution kind,
final requested time, maximum accepted dt, every elapsed-time/dt closure and
requested sample-time stop must agree. Reconstruction and temperature-coordinate
metadata must match the generated inputs. Format consistency cannot authenticate
an arbitrary executable. The calling test runner's
already-open adapter.log is allowed. It is never overwritten by the adapter.

It creates genuine OpenFOAM dictionaries and blockMeshDict for each requested
resolution/variant. Evolution/remap cases invoke real blockMesh and the supplied
real CHMT executable. Local CUDA film contracts need real dictionaries but no
unrelated volume mesh. All commands and complete logs remain under `raw/`.
No successful run.json is written after a solver or output-validation failure.

`observations.csv` is aggregated from raw solver observations. Its keys are
checked independently of the reference values. Each subrun retains actual
step counts, step size, build/device identity, inputs and output. The top-level
step count is the sum over subruns; its time_step is the maximum recorded
accepted step size. Full per-run values and steps.csv histories are authoritative. Source-output SHA-256
is a deterministic digest of actual files inside the raw chmtOutput directories;
`raw-output-manifest.json` also preserves hashes of input/command/log files.
Adapter source bytes and its own actual checkout/build-sidecar revision are
recorded separately from the solver revision.

For preselected temporal refinement, prepare another fresh directory and invoke
the adapter directly with `--dt-scale 0.5`, then use the unchanged package's
comparison commands on that result. The adapter never chooses dt from observed
errors. A baseline run does not claim the separate time-refinement gate passed.

## Model dictionary and physical inputs

`CHMT -case PATH` reads `constant/chmtProperties`, standard controlDict and the
actual default-region mesh. An optional `solidRegion` names a separate actual
OpenFOAM mesh. Dictionary parsing uses Foundation APIs, not a third-party JSON
parser. Text values accept ordinary OpenFOAM word tokens or quoted strings.
Production extensions are explicit data; no string-evaluated cell code.

Required identification/control entries:

- schemaVersion 1; modelName; materialSource; mechanismSource
- modelFingerprint: 64-character lowercase SHA-256 of the frozen configuration
- species: exactly Ns names in compiled order; condensedNames: exactly two names
- elements: up to eight element names, in the declared table order
- minDt, maxDt, cfl; optional spatialOrder (default 2)
- operation: evolve, remap1D, filmProfileContract, filmPressureContract or filmPhaseContract
- enableGas, enableFilm, enableReactions, enableParticles, enableSst, enableRadiation
- filmThermalMode: ThicknessAveraged or ResolvedNormal
- reconstruction: LimitedLinear (production default) or SmoothVerification

The application reads `speciesThermo/<species>` entries R, cp0, cp1, e0, Tmin,
Tmax and element. Species energy is e0+(cp0-R)T+0.5*cp1*T²; element entries are
mol/kg. `condensedThermo/<name>` and `liquid` contain rho, cp0, cp1, e0,
conductivity, Tmin, Tmax and element, with energy e0+cp0*T+0.5*cp1*T².
These are explicit data, not implicitly selected alumina/carbon properties.

Transport entries include gasViscosity, gasConductivity, gasDiffusivity[Ns],
permeability, poreViscosity, liquidViscosity and liquidReferencePressure.
Other inputs are meltTemperature, gravity, emissivity and ambientTemperature.
The optional tolerances dictionary names the absolute/relative mass, energy,
temperature and geometry tolerances plus maxCouplingIterations/maxRetries.

Each reaction dictionary has A, temperaturePower, activationEnergy [J/mol],
condensedNu[2], gasNu[Ns] [kg/mol reaction] and order[2]. The material dictionary
contains phaseCondensed, phaseFilmY, porosity limits, enableMelting,
enableSurfaceReactions, enablePoreOutflow, contact resistances and evaporation
selection/data. Surface reactions additionally supply gasOrder[Ns]. The
backend validates stoichiometry, relevant thermodynamic tables and compatibility.
No reaction heat is added independently of formation-inclusive e0.

The optional sst dictionary overrides named SST coefficients from the shared
SstConfig. SST is integrate-to-wall; initial k/omega fields and wall geometry
are required when enabled. Gas total energy excludes modeled k. The particle
policy specifies explicit condensedIndex, transfer coefficients, contact
mode/duration and emissivity. A particles dictionary supplies position,
velocity, mass, thermalEnergy, diameter, cell/contact indices, contact state,
and quoted decimal 64-bit id/rng. Particle thermalEnergy excludes kinetic
energy. A fresh particle case must explicitly select `particle { volumeMode
DilutePoint; ... }`. The importer then constructs one void-fraction value of 1
per actual gas cell. Optional voidFractionValues supplies values in imported
cell order, but every value must be exactly 1; occupied-volume/dense values fail
before allocation. contactRelativeVelocity/contactMechanicalEnergy are separate
contact-state inputs, never folded into thermalEnergy. Runtime's first particle instance is dilute and inert; unsupported
dense/collision/material modes remain rejected.

Gas-disabled standalone film or normal configurations have no gas inventory.
Disabling external gas does not waive species data needed by pore gas, reactions
or evaporation. Only truly unused tables can be absent.

## Fields, geometry and initialization

Production gas initialization `initial { kind PrimitiveFields; }` reads standard
rho [kg/m³], U [m/s], T [K], and Y_<compiled-name> dimensionless volume fields,
including actual boundary values. Optional SST reads k and omega. Solid fields
are rho_<condensed-name>, rhoPore_<species>, solidEnergyDensity, porosity and
reactionExtentDensity_<index>; the parser checks dimensions and converts densities
to genuine cell inventories with imported volumes. Independent gasInitial and
filmInitial sections can be used for combined configurations.

Actual polyMesh points, face orientation, owner/neighbour, boundary types,
cyclic partner addressing and persistent face IDs are imported. rebuildGeometry
constructs physical measures and cell-face adjacency. Translation-periodic
patches are supported; rotational/scaled cyclic and unhandled coupled patches
are rejected. Boundary dictionaries can select Slip, NoSlip, Inlet, Outlet,
Interface or Empty. MPI partitioned runtime is not claimed; the frontend is serial.

A surface dictionary identifies its actual patch, standalone status, baseVelocity,
pressure, thickness and Y. Film face/edge geometry comes from patch topology;
periodic seam mapping comes from actual OpenFOAM cyclic point pairs. Standalone
prescribedTopTraction is distinct from computed gas traction. An optional
prescribedPressureGradient is an imposed pressure-drop/body-force-equivalent
input, with its actual external work accounted once; it is not transported
thermodynamic pressure. Coupled faces reject nonzero prescribed driving.

The initial conformal coupled importer requires a separate solidRegion,
surface.patch on that region and surface.gasPatch on the gas region. It uses the same canonical vertex correspondence as coupled mesh motion:
area-weighted base vertex normals/thickness determine the declared offset. Dry
faces must coincide; film top vertices must match that offset with reversed
cyclic topology and one-to-one shared-vertex mapping. Genuine curved offsets
may have unequal top/base areas. Dry interfaces allocate geometry auxiliaries
without inventing a film inventory or requiring unused liquid thermodynamics. Nonconformal map import is not implemented
and is explicitly refused, rather than guessing interpolation weights.

Mathematical SineDensity, DensityStep and FilmSineTemperature initializers use
actual imported cell/face bounds and exact averages, not center samples. They
are restricted to their documented rectangular/constant-property initial data.
FilmSineTemperature checks actual four-corner, axis-aligned rectangle geometry
and a tangential sine axis; trapezoidal and tilted faces are rejected before
inventory construction.
FilmIntegralFields accepts explicitly supplied mass, true enthalpy and per-species
face integrals in imported patch order. No initializer evaluates a future field.

## Mesh motion and Stefan coordinates

MeshMotion policies are Static, CoupledRecession and PrescribedSinusoidal.
Coupled film/receding-surface configurations select CoupledRecession even when
the supplied initial geometry is static. The prescribed sinusoidal data are
amplitude, spatialWaveNumber, spatialOrigin,
angularFrequency and timeOrigin. Immutable referencePoints X are stored for
restart. At each true stage endpoint:

`x_k(t)=X_k+A_k*sin(k_k*(X_k-origin_k))*sin(omega*(t-timeOrigin))`.

This componentwise model is exactly the unchanged package's custom separable ALE
mesh, not an asserted reproduction of another paper's tables. Swept volumes and
stage areas come from the actual accepted-to-half/full geometry, never an
instantaneous velocity substitute or a volume-residual reconstruction.

The unchanged Stefan case uses mathematical T_melt=0, whereas constitutive
recovery requires positive internal temperatures. Only this reaction/radiation-
disabled mathematical configuration uses an explicit +1 coordinate origin:
internal Tm=1, wall T=2, solid e0=-1, liquid e0=0, cp=1. The invariant enthalpy
and fraction remain exactly the original ones. Export subtracts the recorded
mathematicalTemperatureOffset once. Production Kelvin temperatures are unshifted.

Resolved-normal layer geometry derives from actual cell volumes/coordinates.
The initial exact cut-cell h and liquid fraction are both retained; initial
average T comes from the shared sensible-average recovery. The evolving solver
uses its real conservative normal model. Exported front is liquid volume divided
by actual column area; wall_heat is the backend's cumulative accepted boundary
heat. Neither is reconstructed from a future similarity solution.

## Raw output and remaining gates

Each successful output time retains raw gas/film/normal inventories and a complete
backend checkpoint. Checkpoints carry actual geometry, inventories, budgets,
model/species/build identity, reference geometry and accepted-stage history.
An unrecoverable step returns nonzero and attempts a last-accepted checkpoint.
All requested times are reached with bounded accepted steps; observation labels
use the requested sample time only within an explicitly checked roundoff bound,
while checkpoints retain the actual backend time.

ALE runs retain every actual accepted stage's raw face sweeps/areas and gcl.csv.
The latter contains the genuine signed four directional face sweeps and endpoint
volumes. Every step/stage/cell must be present. The separate GCL comparator and
independent source review remain necessary; exporting a ledger is not a proof.

For film_plug/shear, boundary_work is accepted Budget.supportWork, including the
prescribed top traction and material-base support work. It is not Phi times time.
SmoothVerification is explicitly selected and recorded for ale_wave; its smooth
wave order does not establish production shock-mode convergence.

Remaining gates include real nvcc/device linking, bounds/race checks, all frozen
field/budget comparisons, space/time refinement, stage-GCL checks, coupled motion
and restart-versus-continuous numerical equivalence. No engineering prediction,
full legacy particle/radiation parity, Freno/TACOT validation, or full design
completion is claimed by these software checks.
