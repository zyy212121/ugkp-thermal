# Reacting wall geometry and source quadrature

`common/gasWall/WallGeometryBuilder.H` is a host-only adapter-neutral builder.
The boundary-layer SST adapters opt into the profile-aware rule by setting
`profileNodes` and `profileStretch` to the actual BVP configuration. Other wall
families do not enter this builder, and `profileNodes=0` retains its geometric
four-point tetrahedral rule for non-SST consumers and geometry references.

## Actual owner volume

The builder triangulates simple planar faces, certifies a strict star-kernel
point, and decomposes the closed owner boundary into positive tetrahedra. It
checks their total volume and first moments against the supplied current-stage
owner volume and volume centroid. A supporting physical wall, exactly one
physical wall for the owner, and a connected matching ray beyond the owner's
complete normal projection remain required. A bounded kernel search prefers a
well-separated seed to avoid numerically flat tetrahedra at concave reflex
planes, with a strict-certificate fallback for narrow kernels.

The quadrature never adds neighboring volume or substitutes wall area times a
one-dimensional matching-layer integral for the real owner volume.

## Profile-aware positive normal measure

For each tetrahedron, intersection with a plane at wall-normal coordinate `y`
is a triangle or quadrilateral. Its area is piecewise quadratic between the
projected vertex coordinates; its area-weighted xyz first moment is piecewise
cubic. The builder combines these areas and first moments at common cuts made
from all projected tetrahedron vertices and all actual BVP nodes:

`y_i = matchingDistance * pow(i / (profileNodes - 1), profileStretch)`.

Above the first BVP interval, the core reconstructs its full solved nodal SST
source linearly. Two positive Gauss points per cut interval therefore integrate
the product of that reconstruction and the true section area exactly, up to
floating-point error. This is exactness for the stated reconstruction, not a
claim that the physical SST profile is exact or BVP-mesh-converged.

The first interval uses the physical near-wall asymptotic state instead of a
linear source. A positive 8-point Gauss rule in `eta = sqrt(y / y_1)` is the
default; 4 and 16 points are supported for quadrature convergence checks. The
mapping regularizes the zero-blowing fractional-power endpoint without
sampling the singular wall. Nonlinear source accuracy still requires a
convergence check; polynomial exactness is not claimed for this interval.

Only distances and positive volume weights are sent to the core. Associated
`quadraturePoints` in a profile-aware descriptor are area centroids used for
xyz moment diagnostics. A centroid may be outside a concave or disconnected
section. `quadraturePointsAreInterior=false` explicitly records this: it must
not be used for point-in-cell claims. Every scalar quadrature weight is
nevertheless obtained solely from positive sections of certified owner
tetrahedra. The legacy tetrahedral points retain their interior certificate.

## Cost and cache identity

Let `T` be the number of boundary tetrahedra, `N` the BVP nodes, and `U` the
number of distinct projected owner vertices plus the kernel seed. Rebuild work
is bounded by `O(T * (N + U))`; it occurs only when geometry or profile settings
change. Resident scalar quadrature storage is `O(N + U)`, not `O(T * N)` and
not exponential in a tetrahedral refinement depth. If `F` cut intervals lie
within the first BVP interval, the number of points is bounded by
`2 * (N + U - 1) + (firstSegmentGaussOrder - 2) * F`.

The cache key includes geometry version, wall selection, profile node count,
stretch, and first-interval Gauss order. Callers must clear it when a rollback
or alternate candidate reuses a geometry version for different coordinates.
Roundoff-coincident geometric cuts are snapped to exact BVP knots so rotated
meshes do not create unsampleable one-ulp intervals.

## Verification

`tests/gas_wall/test_wall_geometry.py` checks cube, skew-prism, tetrahedron,
rotation, concave-star geometry, actual volume and xyz first moments, positive
weights, connected-ray restrictions, rejection atomicity, cache rebuilds,
source integration, first-interval convergence, and bounded scalar storage.

The standalone `tests/gas_wall/test_wall_geometry_profile_probe.cpp` emits nine
JSON records for cube/skew/tetra at 24, 48, and 128 BVP nodes. It compares an
oscillatory piecewise-linear source with an independent polynomial
antiderivative using the analytic actual section area. Each record reports
the integral, reference, absolute error, point count, and fixed acceptance
threshold `2e-11 * (1 + abs(reference))`.

At the tested geometries, source errors were below `5.5e-14`. The 128-node
resident rules used 216 scalar samples for cube/skew and 180 for tetrahedron.
These are manufactured-source geometry results. They do not establish
reacting-wall physical accuracy or replace full BVP and coupled validation.
