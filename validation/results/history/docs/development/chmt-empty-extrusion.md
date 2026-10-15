# Strict-zero Empty trajectories: supported representation

The supported Empty geometry is a two-dimensional cross-section extruded
between two fixed, axis-aligned normal-coordinate layers. Every vertex belongs
to an Empty layer and has one exact tangential-coordinate partner on the other
layer. Each cell has corresponding Empty faces on both layers; each remaining
face projects to one cross-section edge. The normal axis may be x, y, or z.

This first representation explicitly rejects rotated Empty planes, more than
two normal layers, incomplete pairing, and unsupported cell topology. It does
not claim arbitrary rotated three-dimensional Empty support. Non-Empty meshes
retain their existing geometry validity rules and interpolation arithmetic.

## Motion and weighted correction

Both normal coordinates and therefore the extrusion thickness stay fixed.
Paired vertices share the same tangential displacement. Prescribed motion and
correction directions with any nonzero normal component are rejected, including
1e-31: input is not projected because it is small. Paired correction spaces must
be identical/opposite and representable; inconsistent or unprescribed layer
motion is rejected. No nearest-neighbour or tolerance-based pairing is used.

There is one correction coefficient per active layer pair. If the original two
boundary scales are s1 and s2, the common physical scale is
`s = (1/s1^2 + 1/s2^2)^(-1/2)` (evaluated with a scaled formula). Thus a grouped
coefficient q has exactly the original weighted pair objective in real
arithmetic: `(s*q/s1)^2 + (s*q/s2)^2 = q^2`. This is elimination of the required
paired-motion constraint, not an added interior-motion penalty or a change to
the unconstrained non-Empty QR problem. The existing dense solver and all its
rank/compatibility/size guards remain in place.

## Why the Empty sweep is zero

For an Empty triangle, all initial vertices have the same normal coordinate,
and all endpoint normal coordinates are unchanged. Every linear trajectory
normal displacement is therefore exactly zero. Local difference vectors have
zero normal entries; cross products are purely normal and vertex velocity is
purely tangential. The existing triangle swept-volume dot product consequently
evaluates to zero. No computed sweep is subsequently cleared or clamped.

The same represented trajectory is used by volume samples, orientation/GCL
certification, face sweeps, and coupled surface-edge quadrature. Empty volume
and surface samples use `old + t*(end-old)` so evaluation cannot create a
spurious normal-coordinate change from independently rounded weighted sums.
Old planar-face polynomial constraints still retain all three coefficients.

## Lifecycle, costs, and validation

Pairing is local derived state reconstructed from authoritative mesh points and
topology. It is not a persistent cache and adds no checkpoint schema fields.
Geometry/trajectory calls publish mesh and sweep outputs only after all checks;
unsupported motion and failed correction leave those outputs unchanged.

Empty meshes incur additional CPU work: layer classification, an ordered map
for exact vertex pairing, per-cell face correspondence, and linear endpoint
checks. Grouping reduces boundary DOFs but does not remove the original full
geometry/GCL tests. Representation work is reused within one constrained solve;
independent public geometry calls reconstruct it. Surface interpolation has an
Empty-specific path. No new GPU traversal or synchronization is introduced and
no speedup is promised.

Run `bash applications/CHMT/devtools/multirate/check_empty_extrusion.sh` (the
script does not require an executable file mode). It covers coordinate-axis permutations, translated
coordinates, tangential translation/shear/recession, paired correction metric,
coupled gas/solid surface motion, strict-zero sweeps, tiny normal and unpaired
motion rejection, explicit rotated rejection, unaffected rotated non-Empty
geometry, output sentinels, and a real checkpoint write/read followed by the
same trajectory. These are CPU geometry/restart gates, not full GPU endpoint
acceptance or an expansion to arbitrary Empty topology.
