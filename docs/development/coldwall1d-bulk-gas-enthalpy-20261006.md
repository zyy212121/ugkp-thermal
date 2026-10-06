# coldWall1D bulk gas source and enthalpy iteration repair

## Scope

The common eight-node `coldWall1D` / `solidifyingDeposition` implementation is
shared by FSH and CHT, including both physical precisions. Node 0 remains the
wall-contact node. Internal conduction, contact geometry, wall resistance,
phase properties, pinning, the eight-lane tridiagonal layout, and iteration
limits are retained.

The Ranz–Marshall conductance still uses the existing gas snapshot, diameter,
zero stuck-particle velocity, material mass, and density normalization. It is
calculated by lane 0 and broadcast. It is no longer an exclusive node-7 gas
boundary. The Eulerian cell-moment gas-energy closure is unchanged; this repair
does **not** establish an exactly paired particle/gas energy ledger. Mobile
particle relaxation and the separate explicit `coldWall2D` model are unchanged.
Consequently, 2D retains its previous top-surface gas-boundary interpretation.

## One whole-particle exchange per physical step

For equal node masses `m/8`, form `h_bulk = sum(h_old,j)/8` and invert the existing
phase-aware enthalpy relation to obtain `T_bulk`. With the existing whole-particle
conductance `G`, the initial heat-transfer power is `G*(T_gas - T_bulk)`.

A frozen-power Euler increment can overshoot gas equilibrium at large
`G*dt/(m*cp)`. Gas Courant control does not guarantee a thermal-rate bound.
Instead, integrate the bulk source using a frozen **secant enthalpy capacity**:

```
delta_h_eq = h(T_gas) - h_bulk
cp_sec = delta_h_eq / (T_gas - T_bulk)
delta_h_gas = delta_h_eq * [-expm1(-G*dt/(m*cp_sec))]
```

Near equilibrium, use apparent cp instead of an ill-conditioned secant; the
tolerance includes arithmetic precision and the enthalpy inverse's existing
40-bisection resolution. Zero conductance, zero duration, and equal enthalpy
give zero exchange. Non-finite inputs or an invalid capacity are rejected.

This is the usual exact exponential relaxation for constant cp, includes the
existing latent-heat enthalpy in the equilibrium target, and has the correct
small-step limit `delta_h_gas/dt = G*(T_gas - T_bulk)/m`. Across variable cp or
phase transitions it is a first-order frozen-capacity approximation, **not**
the exact nonlinear relaxation solution. Timestep refinement is tested.

The source is evaluated once before the nonlinear loop. Every node receives
the same `delta_h_gas`, hence energy `(m/8)*delta_h_gas`; the sum is exactly the
one whole-particle exchange, up to floating-point rounding. There is no
per-node Ranz–Marshall evaluation, per-iteration repeated heat injection, new
kernel, dense matrix, or extra tridiagonal solve.

## Correct old-time-level storage term

For each nonlinear iterate, `h_guess` is the current candidate enthalpy and
`T_guess` its inverse. Let `C_j = m_j*cp_app(T_guess,j)/dt`. The storage/source
right-hand side is now

```
C_j*T_guess,j + (m_j/dt)*(h_old,j - h_guess,j)
              + (m_j/dt)*delta_h_gas
```

with the existing wall RHS at node 0. The gas diagonal and top-node gas RHS are
removed, including the gas contribution to the stabilized FP32 PCR row sum.
The accepted update is

```
h_candidate,j = h_old,j + delta_h_gas
               + dt*(internal_power,j - wall_power,j)/m_j
```

The formerly missing `h_old - h_guess` term makes all iterations solve the same
implicit storage equation. For constant cp/conductivity, one, four, and twelve
iterations now agree with an independent backward-Euler solution. Nonlinear
phase cases still require convergence: the existing default four iterations
and maximum twelve are unchanged, and four iterations need not converge for
large steps that cross a phase boundary. This repair does not add a nonlinear
convergence guarantee or an adaptive timestep policy.

## Admissibility and precision

The bulk gas increment is bounded by the bulk equilibrium enthalpy difference.
This does **not** guarantee admissibility of every node in a highly stratified
profile: distributing the same negative increment can exhaust a cold node's
enthalpy. Such a candidate is rejected without publishing it; individual nodes
are not clipped. The coupled wall/conduction/phase solve is not claimed to be
unconditionally stable or accurate. Extremely stiff internal diffusion can
also amplify FP32 solve-to-flux cancellation. The existing material-property
and inverse-enthalpy domains have not been extended.

The scalar reference now rejects the same negative/non-finite profile inputs
as the GPU preparation path, before publication. Invalid-input and invalid
cooling rollback are covered by regressions.

## Verification

`tests/test_cold_wall_1d_energy.py` builds and executes the real shared host
helper in FP32 and FP64, covering:

- Independent dense backward-Euler reference, iterations 1/4/12 and multiple dt
- Variable-cp and phase-region implicit enthalpy residuals
- Uniform gas-only heating/cooling and no manufactured nodal gradient, including
  gas stiffness up to 100 and constant-cp exponential equivalence
- Latent enthalpy, nonuniform bulk-source plus wall-energy accounting, disabled
  gas, zero duration, near-equilibrium endpoints, and timestep refinement
- Invalid state/source rejection without mutation
- C++ syntax of the actual shared precision wrappers and source wiring

The syntax checks declare CUDA intrinsics; they are **not CUDA execution**.
Actual CUDA operator regressions use corrected scalar/analytic cold-wall
oracles rather than requiring equality with the obsolete frozen algorithm;
unrelated frozen-operator checks are retained. Native GPU validation still
requires nvcc and a compatible device. No full CFD result or performance claim
is implied by these isolated tests.
