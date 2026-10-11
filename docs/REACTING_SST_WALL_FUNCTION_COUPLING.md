# Inventory-based reacting SST numerical wall function

This change applies only to selected `boundaryLayer/reactingSst` walls. Ordinary
`lowRe` and `wallFunction` branches retain their existing equations and floor
policy. It is a high-Re modeled wall relation, not a resolved-wall k=0 boundary.

The auxiliary, donor-matched U/T/Y/k/omega boundary-value problem supplies wall
traction, heat/species exchange, and a finite omega relation at the owner centre.
Its k profile is a local wall model; it is not a second transported coarse-cell
inventory. Its k source integral and physical low-Re k flux remain diagnostics
and are deliberately absent from the coarse closure record.

For the sole transported inventory Qk=V*rho*k, the owner source is

    P = min[ |tau_t| Cmu^(1/4) sqrt(k)/(rho kappa y), c1 betaStar k omega ]
    Sk = rho [ P - (2/3) div(U) k - betaStar k omega ]

Here k, rho, omega are the current coarse state, omega satisfies the single
numerical-wall owner relation, and y is the geometric owner distance. The coarse
eddy viscosity is still the SST function of that same k, omega and coarse strain.
The ordinary shared internal-face fluxes remain the only internal transport.
The wall uses zero normal k diffusion. Existing nonnegative material blowing
injects zero TKE, matching the stated k_wall=0 material assumption; it never
advects the auxiliary matching k. The constrained omega has no additional wall
injection/diffusion. Its replaced coarse source/internal transport and algebraic
exchange remain distinct in the existing audit.

The donor-matched profile need not integrate to Qk: it is a wall response, not a
claimed conservative reconstruction. This changes the former coupling model.
It does not prove accuracy for arbitrary lift-off, transient turbulence or
blowing; independent resolved-reference errors must be reported. In particular,
zero k diffusion is the high-Re wall-function relation and must not be described
as resolved low-Re behavior. The original auxiliary k source must never be
reintroduced as a physical owner-volume source.

## Source and timestep

Both advance and prediction evaluate the same source closure. For selected
owners the timestep measures P+D+|compression|, not the small difference P-D,
so equilibrium cancellation cannot hide explicit stiffness. Existing 0.25
source control, acceptance/rollback and recovery floors are unchanged. No new
negative-inventory stop or k floor policy is introduced. Exact zero has a
well-defined constitutive limit; normal configured SST floors still apply.

The stage record grows from three to four scalars (one Real per selected face).
There are no additional launches, full-field passes, device/host transfers,
wall-Newton solves or persistent subgrid inventory. Source activity is recovered
algebraically from the already evaluated net source and destruction. Wall
traction uses a tangential projection and max-scaled norm.

## Primary references and exact variants

- OpenFOAM Foundation 10, kqRWallFunction: the documented high-Re k condition is
  a zero-gradient wrapper:
  https://cpp.openfoam.org/v10/kqRWallFunctionFvPatchField_8H_source.html
- OpenCFD 2412, omegaWallFunction, calculate(), source lines 335–340: the
  production relation is (nut+nu)|snGrad(U)| Cmu^(1/4)sqrt(k)/(kappa*y). Replacing
  rho(nut+nu)|snGrad(U)| with the numerical tangential traction gives the formula
  above. This is the exact source lineage of this selected branch:
  https://api.openfoam.com/2412/omegaWallFunctionFvPatchScalarField_8C_source.html
- Foundation 10 omegaWallFunction, lines 255–273 and manipulateMatrix():
  supports the single owner omega algebraic row, but has a different logarithmic
  production expression. The new selected branch is not claimed identical to
  that Foundation production variant; the ordinary branch is unchanged:
  https://cpp.openfoam.org/v10/omegaWallFunctionFvPatchScalarField_8C_source.html
- NASA Turbulence Modeling Resource, SST: resolved smooth-wall k=0 and the
  wall-omega treatment are distinct from high-Re wall functions:
  https://turbmodels.larc.nasa.gov/sst.html

The tests execute production operators in host emulation. They are not CUDA
runtime evidence. The captured owner32 six-face runner explicitly freezes the
surrounding gas/donor boundary data and must not be called a full CHMT endpoint.

## Optional owner-source diagnostic cost

The auxiliary owner-volume integral is now opt-in. Production resident and material solves never request
this integral; direct BVP diagnostic tests explicitly set `WallInput::integrateOwnerKSource`. Disabled calls perform zero owner
quadrature accesses (46 removed point searches for captured owner32), leave the
physical trace, heat, traction, omega and wall flux bitwise unchanged, and emit
NaN in the existing auxiliary source CSV column. This input flag is not a new
model configuration or restart field. Coarse SST budget auditing is independent of
that auxiliary source diagnostic. Geometry validity and the owner volume used by coarse
conservation remain required. The profile-node species/reaction diagnostic loop
is unchanged; this change removes the now-unnecessary owner-k integral only.

Production geometry adapters also disable diagnostic quadrature construction.
The builder still certifies positive-tetra volume and first moments and physical
owner/matching distances, then returns empty point arrays. Zero-length IPC and
null arrays are supported without allocation/upload or wire-layout changes.
Moving geometry revalidates physical geometry on version changes; profile-node
changes no longer invalidate geometry merely for a discarded source integral.
Direct geometry research calls may still request the full quadrature rule.
