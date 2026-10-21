# PR22 mathematical qualification and serial relative efficiency

This directory publishes the frozen scalar reference as three ordered archive
parts, plus separately reviewed shared/CUDA-port evidence. See [REPRODUCE.md](REPRODUCE.md)
for relative reassembly and validation commands. No unsplit copy is stored here.

## Main result

The 24-node, binary equal-diffusivity, no-blow reference completed the unchanged
200-microsecond fixture with **672,000 / 672,000 successful roots**. Maximum
normalized residual was 1.00999545e-8 under the original 1e-8 + 1e-10 criterion;
maximum conservative-moment residual was 8.91623e-15. Neither node count,
timestep, iteration budget nor physical residual tolerance was relaxed.

Two demonstrated defects are corrected:

- The impermeable wall no longer imposes an additional matching-composition
  Dirichlet value. A positive, conservative two-moment reconstruction permits
  the wall composition to respond while preserving zero wall species flux and
  the common physical interface species/enthalpy flux.
- Near an SST branch switch, collective finite-difference shrinking could erase
  healthy Jacobian columns. Admissible mirrored and, when necessary, independent
  scalar directional probes repair derivative construction without changing the
  physical residual, line search or publication.

The exact uniform-pure identities are covered. General unequal-diffusivity,
blowing/reacting and mixed-interior/pure-matching reconstruction paths retain
legacy behavior; they are not silently claimed mathematically corrected here.
No new input gate or state rollback was introduced.

## Accuracy evidence and remaining differences

The 38 manufactured species/energy controls are finite, retain moments to
1.11e-15 and have exactly zero wall species flux. The two-million-row coupled
publication audit verifies the common species flux and its enthalpy contribution.
The original failed root capture remains in the package alongside the repaired
replay; the final live trajectory also resolves that root.

The independent equilibrium controls show a few-percent decomposed thermal and
shear difference from the wall function. They use a uniform mixture and are not
nonzero-species-transport tests. The coupled nonzero-transport fixture still has
heat area-MAE **153.266 W/m²**, 47 opposite heat signs, and a large entrance gap:
owner 0 is **2970.31 versus -126.84 W/m²** on the independently evolved wall-function
trajectory. The absolute heat discrepancy divided by integrated absolute
conduction/recovery/traction-work activity is **2.15855%**, not a net-heat error
ratio. No wall was excluded and no coefficient was tuned to the wall function.
Finite-window transients and reconstruction/closure differences remain explicit;
this is not a declaration of steady state or universal wall-law equivalence.

## Serial relative efficiency

Four serial processes in fixed HEAD/final/final/HEAD order each replay the same
4,000 chronological wall inputs. Every solve and fit succeeds. The final/HEAD
pooled measured wall-operator-kernel time is **1.00732×**; the two paired ratios
are **0.98047× and 1.03454×**. Cost is comparable within this observed spread, not
a demonstrated speedup. BVP and reconstruction separately cost 1.25429× and
1.22549×, while native source/reference-mean/face work costs 0.53471×. Native HEAD
uses four source subdivisions and the coherent final route one; the mathematics
and boundary treatment are intentionally different despite common FP64 inputs,
24 nodes and tolerances. This is not an equal-discretization benchmark.

The timer excludes FV assembly, geometry/store, I/O and failure-capture copying.
No CPU parallel/cooperative tests or GPU timing are used. F/J counters have
different initializer semantics and are not converted into speed ratios.

## Reproducibility

`REPRODUCE.md` gives archive verification and bundle-local scalar-run commands. The relocated
build reproduces the pinned coupled executable byte-for-byte; all 38 manufactured
rows and captured replay states also match their saved reference. `audit.py`
independently reads the included 672,000 raw terminal root records.

The detailed report, exact source pins, raw timing rows and independent mathematical
review remain under `source-sign-audit/math-gap-20261021T0626Z/`. The package
manifest verifies every included file. CUDA execution and the user's GPU review
remain separate validation steps.

## Shared/CUDA-port evidence

The exact reviewed `shared-port/shared-math-port-evidence-20261021.tar.gz` contains
98 hash-verified evidence files for source commit
`b5f92ef90472c8badf23118633557230a0e75ddd`: targeted scalar regressions, six
byte-identical frozen-profile comparisons, current shared scalar driver results,
relocated-checkout checks, source pins and the patch. Its complete scope is in
the [shared-port report](../affine-directional-shared-port-20261021.md).
The device tests use current shared `SerialWallTeam` oracles, not a stale
external profile header. **nvcc, GPU execution and CUDA sanitizers are UNRUN**;
the user performs device review and validation. No GPU acceleration is claimed.
