# Reacting SST owner coupling: reproducible verification

## What changed and what this evidence establishes

The production change uses a high-Re numerical wall function for the selected
`boundaryLayer/reactingSst` branch. Auxiliary wall traction and finite owner
omega close the actual coarse k equation. The independent auxiliary k profile
is no longer treated as the transported owner inventory. See
`docs/REACTING_SST_WALL_FUNCTION_COUPLING.md` for equations, exact OpenCFD versus
Foundation lineage, source/timestep identity, blowing assumptions and costs.
Ordinary lowRe/wallFunction branches retain their existing equations.

The former demand that the donor-matched BVP source be an exact owner-inventory
reconstruction, and the subsequent A-prime moment constraint, were superseded
model recommendations. They are not prerequisites imposed by the user. This
alternative must instead be judged on its actual inventory conservation,
stability, physical model discrepancy and computational cost.

The scripts distinguish four scopes:

1. Actual shared SST operators on captured owner32 and its six incident faces,
   with prescribed surrounding gas/donor data, advanced to 0.2 s.
2. An independent wall-resolved, variable-density 1-D U/T/k/omega SST Couette
   reference, compared with the actual high-Re coarse owner operator.
3. Nonlinear wall solves plus actual coarse owner advancement for three retained
   blowing states, a finite chemistry/heat/blowing state and a zero-blow limit.
4. The unchanged original gasUGKP GPU flatplate case and a separate unchanged
   CHMT moving case, packaged for a real GPU run. These have not run here.

## Reproduce host evidence

From the repository root, with Python, NumPy, SciPy, pytest and g++ available:

```sh
PYTHONDONTWRITEBYTECODE=1 bash research/sst_wall_function/run_checks.sh
```

The scripts compile and call the actual common production C++ operators for
all coarse-owner tests. Only the independent resolved reference has its own
numerical equations. Generated binaries, logs and full profiles are written to
`results/` and are not versioned. `metrics_snapshot.json` preserves the reviewed
small numerical outputs. `acceptance_contract.json` separates exact budget and
convergence checks from physical error reporting; it does not impose a post-hoc
13–16% turbulence-accuracy acceptance threshold.

## Original failure provenance and owner endpoint

The original failure is **gasUGKP flatplate_fixed**, using shared gas/SST code,
not a CHMT runtime. `owner32_capture.json` contains the retained native manifest,
all six actual incident faces, owner/neighbour geometry, old observed values and
source-file hashes. Original binaries:

- solver SHA256: `ea2248c68227f561aa9e853595f1dc80180b888ff336d952aa6c17ef540f859f`
- backend SHA256: `02b31f879903ab5ba4d96232f8e2ba209e1985ee89ef91384fdd7faeae166ecf`

The original native diagnostic run exited naturally with SourceStepLimit before
0.2 s. Its 228 diagnostic records are not 228 accepted time steps. The last
observed dt was 1.45313386258e-14 s and source number 0.32128. The auxiliary
profile Qk was about 267960 times the actual initial owner Qk; its source
integral was -9.45644541e-4, imposing roughly -3693.49 per unit owner volume.

The new replay retains p=101325 Pa, T=300 K, U=70 m/s, matching k=1e-4,
matching omega=1, owner distance 0.000614471929902 m, matching distance
0.00187121251709 m and actual owner volume 2.56029970793e-7 m3. It advances only
the actual owner's conservative SST state; the donor and bulk gas remain fixed.
Each step repeats the real nonlinear wall solve, publication, gradients,
eddy viscosity, source prediction, all six SST face fluxes, source update and
recovery. The outer maximum step 1e-5 s and target Courant number .3 are read
directly from the retained original controlDict. No source threshold, physical
k floor or endpoint was relaxed.

Measured endpoint results are recorded in `metrics_snapshot.json`:

- endpoint 0.2 s; 20032 steps and wall solves, 100160 Newton iterations (89.25 s host time)
- minimum dt 4.54192744715e-6 s; maximum applied source number 0.223496173807
  below the unchanged 0.25 guard
- minimum k=1e-4; final k=5.58383907385; no recovery floor activation
- actual source impulse +0.00110803290373 and face impulse -0.00110640684912
- global k budget residual -2.17e-19; maximum per-step residual 1.23e-22

This demonstrates removal of the inventory/source-collapse mechanism in the
captured control volume. It is not a full 4608-cell gas run, a moving CHMT run,
or proof of the fixed-donor assumption's physical accuracy. The old full native
run and the new fixed-boundary replay are not comparable throughput workloads.
Host time in the snapshot is a reproducibility measurement, not GPU throughput.

## Independent nonzero SST resolved reference

`resolved_sst_reference.py` independently discretizes steady 1-D mass-consistent
variable-density U, thermal energy, k and omega equations with full SST F1/F2,
production limiter, cross diffusion and viscous heating. It does not import a
production SST leaf. Density is p/(R*T). Inputs are L=.01 m, Uedge=30 m/s,
Tw=Tedge=500 K, kedge=.5 m2/s2, omegaedge=200 1/s, p=101325 Pa,
mu=2e-5 Pa s, Cp=1040 J/(kg K), molar mass .028 kg/mol, Pr=.72 and Prt=.9.
Wall k=0 and finite resolved wall omega=60*nu/(beta1*y1^2) use y1=1e-7 m.
The exact resolved-wall specification follows the
[NASA SST reference](https://turbmodels.larc.nasa.gov/sst.html).

Geometrically spaced grids contain 128, 256, 512, 1024, 2048, 4096 and 8192
nodes. Picard iteration uses positive implicit destruction and thermal excess
temperature to avoid subtractive error at 500 K. The stopping test is relative
update <2e-10; separately recomputed final-state FV residuals are normalized by
gross face/source activity and are <1.1e-9 in every equation/grid. This is a
nonlinear/discrete convergence check, not an accuracy certificate.

| nodes | wall shear Pa | conductive heat W/m2 | cell-average k at y=.00075 m |
| ---: | ---: | ---: | ---: |
| 512 | 1.040818567 | -16.31026254 | 3.818741780 |
| 1024 | 1.040909495 | -16.29994831 | 3.818673670 |
| 2048 | 1.038873283 | -16.26181824 | 3.810148604 |
| 4096 | 1.039370950 | -16.26705612 | 3.812192814 |
| 8192 | 1.039785081 | -16.27218876 | 3.813913315 |

Refinement is mildly nonmonotone around SST switching functions; 1024 alone
was therefore not treated as definitive. From 4096 to 8192, shear changes
0.0398%, heat 0.0315%, and the three compared density-weighted owner-average k
values change 0.0430–0.0451%. All grid results and all three cell averages are
retained, rather than selecting only the most favorable grid difference.

`compare_couette.py` obtains coarse rho/U/T and reference k by density-weighted
integration over the physical owner slab [0,2y], samples the independently
resolved donor at 3y, then drives the actual coarse production/source/transport
operator to its nonzero steady state from k_reference/20. It does not seed the
answer or compare a duplicated production formula to itself. The wall solve
uses 128 nodes. The normalized coarse source-plus-flux residual is <1e-9.

| y m | y+ | matching m | shear error | heat error | omega error | owner-average k error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| .00075 | 31.589 | .00225 | +0.03848% | +0.05052% | +0.01020% | -13.3806% |
| .00150 | 63.178 | .00450 | +0.09466% | +0.10434% | -0.02914% | -12.3475% |
| .00225 | 94.768 | .00675 | +0.15565% | +0.16341% | -0.01454% | -16.1669% |

The shear/heat/omega agreement and finite conserved turbulent equilibrium are
useful evidence. The 12–17% coarse k deficit is a real high-Re closure/model
error and remains visible for review. It is not converted into a universal
physical PASS. This resolved accuracy comparison has zero blowing; the next
section verifies coupling and limits with blowing, not resolved blowing accuracy.

## Blowing, chemistry, and actual owner budgets

`run_blowing.py` retains Uedge=30, Tw=Tedge=500, kedge=.5, omegaedge=200,
owner y=.004 and matching y=.01, with m=0, 1e-9 and .001 kg/(m2 s), at both
32 and 100 profile nodes. Its original owner quadrature is two points at .002
and .006 m with weights 2e-7 m3 each. Area is 5e-5 m2; owner volume 4e-7 m3.
Owner U=25 and k=.5 initially; donor/bulk gas are prescribed. A compensating
external mass outflow holds bulk mass fixed. Actual published wall mass and
energy pass through the shared gas wall-flux consumer, and actual k/omega face
fluxes, source prediction and owner advancement execute on every step.

A fourth state includes simultaneous reaction and heat: Tw=500, Tedge=700,
A->B rate20, species A formation enthalpy100000 J/kg, diffusivities 1e-4/3e-4,
and the same m=.001 split .7/.3 between species. It is not a zero-chemistry
placeholder. Additional m=1e-12 and 1e-15 cases test the zero-blow limit.

Every state reaches .005 s in 250 steps, with no floor activation. Maximum
source number is about .02138. k source, transport and rounding constraint
channels agree with a separately accumulated budget. Maximum global k defect
is <5e-22; omega transport/source/algebraic-constraint balance is <7e-18.
Mass, tangential momentum and physical total-energy publication close to
roundoff; maximum energy-rate defect is 3.5e-18 W. Auxiliary chemistry balance
is <6e-16 kg/(m2 s), with nonzero reaction integral approximately -.0511/-.0526.
The wall injects zero TKE by the documented physical material assumption;
physical omega wall flux is zero because the owner row is algebraic. Nonzero
production, internal transport and omega constraint impulses are retained.

At m=1e-9 the shear differs from zero blowing by approximately 1.5–1.7e-5
relative. Reducing m to 1e-12 and 1e-15 decreases the difference monotonically,
approximately as sqrt(m); the final relative shear difference is about 1.7e-8.
This explicitly checks a continuous limit instead of equating tiny-blow and
zero-blow solutions by clipping their input.

## Cost and optional diagnostics

Production gains no additional full-field pass, GPU launch or BVP solve.
The coarse closure gains one Real per selected face. The optional diagnostic
input bool adds 8 bytes after alignment in FP64 (Ns2 input 504->512 bytes,
Ns10 input 696->704 bytes), with no new independent array or restart/configuration
field. A disabled owner-source diagnostic removes the captured case's 46
quadrature searches per BVP, including near-wall source reevaluations. Tests
supply null point arrays on this path and verify FP32/FP64 physical outputs are
bitwise identical to enabled diagnostics. The default owner replay disables
this unused diagnostic; SST audit channels remain available separately.

Existing CSV columns owner_k_source_integral/integratedKSource are auxiliary
source diagnostics only, and are NaN/unavailable if not requested. The existing
wall_k_flux/wallKFlux columns are also auxiliary low-Re profile diagnostics,
not the physical coarse k exchange. Actual coarse budgets use SST audit
channels. Production geometry adapters also skip diagnostic quadrature construction,
allocation and upload. `common/gasWall/WallGeometryBuilder.H`,
`wallGeometryDetail::build`, still certifies positive-tetra physical volume,
first moments, owner distance and unconstrained matching donor. Its internal
construction switch returns exact certified geometric moments and empty point
arrays instead of entering `profileOwnerQuadrature`. Direct geometry research
calls can still request that rule; no new user dictionary or restart field is
introduced.

The removed rule sorted projected cuts and scanned intersecting tetrahedra at
every quadrature point: approximately O((T+N)log(T+N)+Q*T), for T tetrahedra and
Q points. Three host arrays required 5*sizeof(Real)*Q payload bytes, and device
distance/weight arrays required 2*sizeof(Real)*Q. For captured Q=46, production
now avoids 1840 host payload bytes and 736 device bytes per wall plus vector
allocation overhead, as well as the corresponding transfer on geometry rebuild.
These are exact storage/loop counts, not a measured GPU speedup.

`WallClosureHost::prepareGeometry` and
`SharedGasDeviceStorage::prepareWallGeometry` retain geometry-version caching
and physical recertification on moving-mesh rebuild. BVP nodes/stretch no longer
invalidate that geometry. Gas `BoundaryLayerInput` and
`private_backend/BoundaryLayerAdapter.cuh` send/accept zero-count CSR quadrature
with null arrays. The 80-byte IPC configuration layout is unchanged. Tests cover
actual zero-length client socket payload, backend no-allocation behavior,
CHMT initialization/moving rebuild, and continued rejection of bad volume and
cell-centroid moments. The source-diagnostic switch is independent of actual
coarse inventory audits, which never needed this profile integral.


## Retained full GPU cases

Both retained runtime cases use two gas species and require matching Ns2
binaries. The CHMT MSS7 material card is explicitly controlled-synthetic, not
validated physical ablator data. Input manifests preserve all original
0/constant/system files. Preparation
verifies their hashes and creates a new output directory without altering the
retained inputs. It returns NOT_RUN. On a machine with the final rebuilt
binaries, compatible CUDA driver and GPU, run:

```sh
python3 research/sst_wall_function/run_gpu_cases.py \
  --case gas_flatplate --output /absolute/new/gas-result \
  --solver /absolute/path/gasUGKP --backend /absolute/path/gasUGKPCudaBackend --execute
python3 research/sst_wall_function/run_gpu_cases.py \
  --case chmt_mss7_moving --output /absolute/new/chmt-result \
  --solver /absolute/path/CHMT --execute
```

Source the binaries' OpenFOAM/CUDA runtime environment first. The gas case must
reach its original .2 s and write the actual fields; the separate CHMT case must
reach its original .01 s with accepted-step CSV evidence. The runner waits for
natural process termination without shortening endpoints or reporting a stopped
run as success. It records input and binary hashes, command, exit code and
terminal evidence. PASS_ENDPOINT_ONLY does not imply accuracy/budget/restart
acceptance; those require reviewing the solver's actual diagnostics. No GPU
runtime, full-case endpoint or GPU throughput result is claimed here.
