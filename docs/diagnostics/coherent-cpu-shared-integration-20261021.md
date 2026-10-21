# Provisional coherent wall integration — CPU evidence, 2026-10-21

## Scope

This isolated change is based on `2f4a66f82c8405fad65542490c4d88841b2b9cd4`.
It integrates the qualified generalized **Ns=2, FP64 conservative-column**
operator: coherent Hermite reconstruction and reduced-energy identity, cached
banded LU, tie-aware PCHIP finite differences, and common physical matching
T/Y. Matching p/U/k/omega and the ordinary gas Riemann operator remain unchanged.
The selected source quadrature is one subdivision; legacy reconstruction retains
four. Default production resolution remains 24 nodes.

The same storage-route predicate selects matching, solve and reconstruction.
Other species counts, non-FP64, analytic, uncovered/nonconservative and no-coherent-
storage routes retain their existing solver/representation. Nonzero wall mass,
species flux and chemical terms remain in the generalized equations; there is no
new physical-input rejection, tolerance relaxation or iteration-budget change.

## Ownership and continuation

- Worker scratch is pooled and separate from each wall's numerical state/LU.
- Accepted macro history remains separate and is committed only at its existing
  boundary. Numerical warm-state reuse has a separate diagnostic counter.
- A finite physical publication can be nonconverged. The returned bool still
  describes publication; convergence controls the existing accepted-history
  validity flag, not a new veto of the published candidate.
- Frame changes transform numerical coordinates or invalidate numerical work,
  not the physical calculation. Native warp, pool and split reconstruction
  kernels remain in place.
- Shared scratch points/fluxes are reset on every residual evaluation because
  their constitutive helpers accumulate into those objects. A repeated-F test
  retains this regression.

Current host `sizeof` values for capacities 24 / 48 / 128 are respectively
214,968 / 758,232 / 4,965,272 bytes of new worker scratch and
66,528 / 129,504 / 339,424 bytes of wall-owned numerical state/LU. These are not
GPU occupancy or device-allocation measurements.

## CPU evidence

- Captured warm/cold roots, repeated residual evaluation and all reported physical
  values match the frozen generalized candidate exactly for host teams of
  1, 4, 7 and 32 lanes. Residual/FD node work is cooperative; LU and final
  publication remain leader-owned.
- Two walls sharing one worker reproduce private controls. Advisory candidates
  are published while accepted history stays unchanged and the existing candidate
  validity flag remains false.
- Nine existing generalized cases, including positive/negative net mass flux,
  counterdiffusion, frame velocity changes, SST-disabled and two-node controls,
  plus the existing reacting/frozen pair match the frozen candidate exactly.
  The reacting pair is cold-seed/refresh coverage, not a rebuilt reacting-Jacobian
  proof.
- The same disclosed mixed no-blowing fixture at **20 steps / 4 microseconds**
  reproduces all 17 final fields, 19 initialized/boundary arrays, all 14 physical
  CSV ledgers/audits, 20 non-timing history rows, 13,440 solve headers and selected
  nodal roots byte-for-byte. There are zero failed roots/captures; maximum residual
  is 1.00944394075e-8 under the unchanged 1.01e-8 criterion. All 40,320 same-face
  audit rows agree. This is a host-emulated operator integration test.
- Legacy routes reproduce 52 HEAD-versus-host-team outputs plus five unchanged
  history/profile assertion tests. The full pytest suite was not run in this
  environment.
- Address/undefined-behavior checks pass with leak detection disabled because
  LeakSanitizer cannot operate under the environment's ptrace setup. A four-lane
  ThreadSanitizer check passes. No leak-cleanliness claim is made.
- Capacity 48/128 evidence includes bounded indexing, profile resampling and
  publication-status controls, not converged fine-grid SST trajectory evidence.

The archived audit directory is `source-sign-audit/integration-cpu-shared-20261021`
in the task evidence workspace. It retains red tests, dependency/output manifests,
independent reviews, frozen-reference comparisons and the complete four-microsecond
control. It is not required by the self-contained checkout tests below.

## Reproduce essential tests from this checkout

Requires Python 3 and a C++17 `g++` toolchain with pthreads. These are ordinary
`unittest` tests, also collectable by pytest; they do not install or emulate pytest.

```sh
python3 validation/tests/gas_transport/test_conservative_matching_physical_trace.py
python3 validation/tests/gas_wall/test_coherent_shared_numerics.py
python3 validation/tests/gas_wall/test_coherent_wall_storage.py
python3 validation/tests/gas_wall/test_coherent_production_route.py
python3 validation/tests/gas_wall/test_coherent_components.py
```

`fixtures/coherent_warm_regression.H` contains numeric input/quadrature/iterate
values from the recorded mixed step-538 obstruction, with inherited existing
Air/CO2 material tables. It contains no serialized pointers or external file
paths. The production-route test checks the repaired warm root, repeated F,
pooled history, advisory publication and legacy dispatch at all four host-team
sizes. Component tests cover generic precision/species energy arithmetic,
borrowed profile slopes and the retained capacity/status controls. Test-only
zero/short iteration budgets exercise advisory semantics; production controls
are unchanged.

## Still unqualified

**No nvcc or GPU exists in this environment.** Host/device annotations and CPU
thread collectives do not establish device callability, warp safety, device stack
or register usage, allocation viability, native whole-application behavior or GPU
performance. The next required environment action is an actual CUDA build and
bounded native replay of these same controls, followed by device race/memory checks
and the requested whole-application performance comparison. No merge or remote
publication is implied by this provisional local integration.

This CPU migration proof also does not settle physical agreement or efficiency.
The observed wall-function/BVP differences and unsteady thermal offsets remain
reported differences, not a new acceptance threshold or proof that either model
is an oracle. No longer physical trajectory or coefficient tuning is included.
