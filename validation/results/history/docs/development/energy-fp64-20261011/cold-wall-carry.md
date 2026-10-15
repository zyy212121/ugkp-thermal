# Cold-wall profile mean authority

## Scope and representation

`GPU_PAIRED_PARTICLE_HEAT` selects the CHT precision repair. Without that macro,
1D and 2D retain their existing native-`GpuReal` thermal paths. Persistent nodal
profiles remain float: eight nodes for 1D; eight axial by eight radial nodes for
2D. No double nodal profile or second profile buffer is allocated.

The particle's existing `pThermalEnthalpyResidual` is a double **specific**
enthalpy correction (J/kg). For equal-mass stored nodes `f_j`, the authority is

```
Hmean = sum(f_j)/N + r
Eparticle = represented_mass * Hmean
Heffective_j = double(f_j) + r
```

The correction is uniform. Sub-float-ULP spatial variations are not independently
stored. Their mean is retained, feeds every subsequent node temperature and phase
calculation, and is absorbed into float nodes as it becomes representable.
Double thermal scratch, including the 1D PCR solve and eight local 2D axial
values per lane, prevents losing increments before the final float publication.
Geometry, persistent ring masses, node arrays, and public temperature caches
keep their existing storage types. This adds double arithmetic/register pressure
in the opted-in cold thermal paths; performance has not been measured.

## Uniform gas publication for 1D

`ugkwp::publishColdWallUniformSpecificHeat(nodes,N,r,dh)` validates the old
profile, effective positive nodes, finite candidate values, and positive rounded
float candidates before writing. It publishes

```
f'_j = floor_to_float(double(f_j) + r + dh)
D = sum(double(f'_j) - double(f_j))/N
r' = (r - D) + dh
accepted_dh = D + (r' - r)
```

This delta-form subtraction avoids cancelling large absolute enthalpies. The
residual is the average float-rounding correction, bounded by the mean of the
individual ULPs (plus double arithmetic roundoff); it is not an ever-growing
error account. Zero proposed heat preserves all node and residual bits.

Nearest-float publication is unsafe for highly nonuniform profiles: a negative
mean rounding residual can exceed the smallest node's positive enthalpy. Two
frozen counterexamples cover cooling `[1400.125, 5e6, ...]` to almost zero, and
heating `[0.10001, 4e6, ...]` with old residual `-0.1` by `0.001 J/kg`.
The latter previously changed the effective minimum from `1.00002e-5` to
`-0.085615 J/kg`. Directed publication leaves it positive at `0.133135 J/kg`;
the changed spatial detail is bounded by one profile-storage ULP and the exact
mean increment remains retained. It is not an extra heat source.

Before sharing negative candidate credit, the cooling bound leaves
`16 * DBL_EPSILON * max(effectiveH) + FLT_MIN` above zero. This is a double
arithmetic cancellation margin, not a full float-ULP thermal cutoff. After
computing the actual stored mean and residual, the publisher validates every
actual effective node before any write. Randomized 8/64-node candidates over
`1e-20..1e20 J/kg` cover accepted cooling, heating and both initial carry signs.
The full profile solvers use the same directed storage and publish the validated
raw float directly, avoiding an add-carry/subtract-carry round trip.

The 1D input path first synchronizes all lanes after calculating the original
physical material mass/geometry, calls `publishPairedParticleGasHeat` once in
lane zero, and synchronizes again before loading the updated profile. The old 1D gas conductance
and old bulk source are disabled under the macro. Completed-contact gas-only
handling refreshes phase/history without reapplying or discarding the heat.

## Existing 2D boundary source retained

The existing 2D gas law is **not** the 1D uniform bulk source. It applies
`G/8 * (Tgas - Ttop)` to the top axial node of each radial column at each explicit
substep. The paired generic publisher must skip these 64-node particles.

The 2D wrapper retains that source and multiplies its existing native-precision
conductance by the accepted cell coefficient. If converting the scaled double
conductance to `GpuReal` would round upward, `nextafterf`/`nextafter` rounds it
back down. Each substep accumulates signed `gasPower*stepTime` in double. This
preserves the top boundary condition, conduction stencil, phase law and existing
substep selection rather than replacing them with a uniform gas pre-update.

A safe positive gas reservation is `G * dt * Tgas`, because accepted substeps
have positive temperature. A tighter bound based on the initial minimum of wall
and particle temperature is **not guaranteed**: the existing explicit stability
rule uses local apparent heat capacity, which can fall sharply on crossing a
phase boundary, so it does not establish a temperature maximum principle across
that crossing. The safe bound can undercouple stiff cases. Candidate reservation
must also conservatively cover the native-precision conductance calculation.

## Mean publication and actual wall ledger

For both profile solvers, let `W` be the exact native-precision value ultimately
added to the double wall ledger: `GpuReal(parcelMultiplicity * wallEnergyJ)`.
The accepted specific increment is

```
1D: dH = -double(W)/represented_mass
2D: dH = integrated_gas_energy/physical_mass
          - double(W)/represented_mass
```

The computed candidate profile first receives one uniform correction so its
mean targets the old authoritative mean plus `dH`. Float nodes are published and
the residual is then closed as `(old_r - stored_mean_change) + dH`. Derived
mean temperature and connected phase use the actually published nodes plus the
new residual. Wall physics is unchanged; the tiny mean correction includes the
existing wall-ledger rounding instead of silently depositing that discrepancy
in the gas-particle budget.

The 2D group adds `represented_mass * actual_specific_change + double(W)` to
`particleHeatBudget[cell]`. This is the signed gas component of the actual
accepted particle inventory, with external wall work removed. Zero-duration
updates return without changing profile/residual/history. Internal conduction
cannot inject a net energy drift into the profile mean.

## Legacy ordinary/cold transitions

The ordinary alumina and cold phase-change enthalpy laws differ macroscopically.
This patch does not replace either law or invent an enthalpy-offset state.
Activation and release retain the same legacy reference change, while preserving
the residual unchanged in J/kg. Precision comparisons across those transitions
must subtract the **same legacy transition baseline**; a raw difference between
the two pre-existing enthalpy reference curves is not evidence of a new carry
leak. Cold active nodes use the existing cold law via the common double caloric
adapter, including its original endpoint-trapezoid liquid branch.

## Executable evidence and limits

`tests/gas_transport/test_cold_wall_thermal_carry.py` compiles actual production
1D/2D thermal-group code into an eight-thread, synchronized shuffle/ballot CPU
harness for both Real32 and Real64. It tests:

- 10,000 microincrements into 8- and 64-node nonuniform float profiles;
- exact zero-heat and zero-duration identity, bounded residual, finite cooling,
  rejection without writes, and phase-boundary effective temperature/fraction;
- actual 1D/2D wall-only and nonuniform-profile updates against the published
  wall ledger and physical profile-plus-carry inventory;
- signed 2D top-boundary gas work, a single-substep analytic source check,
  liquidus heating and solidus cooling crossings;
- both complete production wrappers, including a real counted 1D callback and
  bitwise comparison with the same preheat followed by the actual wall solve;
- ordinary/1D/2D shared-cell actual transactions, both signs, near-floor gas,
  nearly-zero cold nodes, and the positive-heat nonuniform counterexample;
- 200,000 legal logarithmic profiles per precision, with no rejected accepted
  candidate and actual heat bounded by double arithmetic roundoff;
- the complete 2D production wrapper at cell coefficients 0, 0.3 and 1,
  checking actual float profile plus carry, double wall ledger and cell budget;
- wall-only updates with null heat-coefficient/budget allocations;
- no duplicate 1D gas source and preservation of carry across actual legacy
  clear/initialization helpers;
- source contracts for the lane-zero 1D publication ordering and 2D cell budget.

The cold wrappers execute controlled long-deposit cases. The 1D callback
counts invocations and delegates all physics to the actual paired publisher.
Mixed tests include the real reduction, acceptance, ordinary/cold publication,
wall ledgers, settlement and gas recovery. They are thermal-only transactions:
transport, collision, pressure, injection and tracking require the separate
complete strictAdvance native fixture. CUDA FMA, atomic ordering, warp launch
behavior and performance are not established by this host lane adapter.

Run from repository root:

```
PYTHONPATH=/workspace/scratch/98b6339ec81b/chmt_implementation_20261011/python_deps \
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/gas_transport/test_cold_wall_thermal_carry.py \
  tests/test_cold_wall_contact_routes.py tests/test_cold_wall_1d_energy.py \
  tests/test_cold_wall_cuda_fixture.py tests/test_cold_wall_launch_default.py -q
```

Verified on 2026-10-11: **179 passed** for that command, including **67** new
cold-carry cases. `git diff --check` passed for the owned files. Test-first
initial execution failed on the missing cold-carry helper before implementation.
