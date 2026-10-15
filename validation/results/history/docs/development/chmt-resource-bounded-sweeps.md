# Resource-bounded swept-volume correction

The production default no longer rejects a required correction merely because
it has more than 512 boundary degrees of freedom, 4096 equations, or 2,000,000
point-mode entries. Those counts were proxies for dense storage, not geometric
validity conditions. The solver now chooses cached or regenerated harmonic
modes using an explicit correction-workspace allocation budget.

## Controls and failure semantics

`SweepConstraintControls::maximumWorkspaceBytes` defaults to 256 MiB and must
be positive. The existing `maximumDegreesOfFreedom`, `maximumConstraints`,
and `maximumModeEntries` default to zero, meaning no additional count limit.
Explicit positive values still impose hard limits. Negative signed limits
are invalid. `maximumModeEntries` remains the logical point-count times
DOF-count limit, even when modes will not all be stored.

As before, an already satisfactory initial trajectory does not need a
correction workspace and does not trigger correction-count limits. It still
passes the complete physical trajectory certification.

Every correction-workspace allocation is checked before calling the underlying
allocator. Counts and byte arithmetic are checked for overflow. Budget failure,
allocation failure, or unrepresentable allocation/index size is an explicit
`SizeLimit` rejection; mesh and sweep outputs are published only on success.
Error text distinguishes workspace exhaustion from an allocation failure.
The algorithm never discards equations or rescales physical targets to fit.

## What the byte budget measures

The solve-local allocator counts requested vector capacities, including outer
row/mode vector objects and allocations for:

- Point weights; boundary and correction vectors.
- Sorted harmonic edges, offsets, inverse-edge weights, weight sums and both
  Jacobi iteration vectors.
- Either all cached modes or one reusable regenerated mode.
- The complete reduced Jacobian, QR right-hand side, correction coefficients
  and row-norm workspace.

The recorded `peakWorkspaceBytes` is the maximum simultaneous live allocation
request total in this domain. It is not process RSS. The caller's mesh and
input/output storage, existing nonlinear points/residual/sweep staging,
`HostMesh` copies, Empty representation and geometry/trajectory certification
temporaries, allocator metadata and retained heap pages are outside this
budget. Whole-process peak RSS must therefore be measured separately.

There is no global allocation hook, shared counter, persistent cache or lock.
Each call owns its budget and harmonic data. An allocation failure unwinds all
of that call's workspace before returning without publishing a partial mesh.

## Algorithm and mathematical meaning

The incident-face-area boundary metric, Empty paired-DOF scaling, analytic
sweep derivatives, normalized rows and all three nonconstant coefficients of
each planar trajectory are retained. The linear correction uses the existing
row-pivoted, twice-reorthogonalized QR of the reduced Jacobian transpose.
Its rank tolerance and dependent-equation compatibility checks are unchanged.
This minimizes the correction norm in the weighted boundary coordinates at
each linearization; it is not a claim of a global nonlinear minimum.

Before allocating modes/J/QR, the solver checks their simultaneous live byte
requirements together with the already allocated harmonic workspace. If the
cached form fits, it generates each mode once and reuses it across Newton
iterations. Otherwise it checks the streamed form: generate each mode to fill
one Jacobian column, release/reuse that mode, run the same QR, then regenerate
modes in the same order to form the point correction. If neither fits, reject.
Actual allocator checks remain active even after the planning check.

The Jacobian remains dense, and QR still modifies its rows in place; there is
no additional dense Q/R copy and no normal-equation solve. This removes the
full point-by-mode storage requirement, not every possible scaling limit.
A Jacobian that cannot fit the configured workspace is explicitly rejected.
No PETSc, CHOLMOD, approximate least-squares fallback or new build dependency
is introduced.

Harmonic adjacency is sorted once in the same neighbour order as the previous
sets. Inverse-edge weights and sums are computed once for this reference
geometry, and Jacobi vectors are reused. The iteration arithmetic and stopping
criterion retain the previous semantics. The public harmonic-motion API is
unchanged; its initial input validation still runs before correction.
The plan is destroyed on return, so movement and restart cannot reuse stale
topology/weights. Supported Empty pairing and its strict-zero normal trajectory
remain the same as documented in `chmt-empty-extrusion.md`.

## Costs and diagnostics

For D modes and K completed Newton corrections, cached mode generation costs D
harmonic solves; streaming costs 2KD. A QR rejection skips the second pass of
that iteration. `modeSolves` reports this work, excluding the initial public
harmonic solve. `streamedModes` records the selected strategy. The streamed
path repeats point-vector initialization/reconstruction, harmonic iterations
and mode accumulation; it does not repeat topology sorting for every mode.
Both paths retain the original residual, line-search and physical trajectory
checks. No new GPU traversal or synchronization is introduced.

This is a memory/time tradeoff, not a general speedup guarantee. The default
strategy keeps the cached path when the complete correction workspace fits.
Whole-window cost also includes existing geometry rebuild and certification.

## Verification

Run the ordinary CPU gate with:

```
bash applications/CHMT/devtools/check_host.sh
```

The focused gate and optional larger default-budget stress case are:

```
bash applications/CHMT/devtools/multirate/check_large_sweep_constraints.sh
bash applications/CHMT/devtools/multirate/check_large_sweep_constraints.sh --stress
```

These exercise all three former default-count failures through the production
entry point, cached/streamed physical equivalence, near-rank compatible and
incompatible targets, multiple Newton iterations, explicit count limits,
budget/invalid-input rejection without output mutation, and all three planar
trajectory coefficients. Existing Empty, moving/remainder and real checkpoint
tests are also required. CPU tests are not a native GPU runtime acceptance.
