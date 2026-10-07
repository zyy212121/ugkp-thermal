# Native OpenFOAM 10 coupling verification

This directory is a **verification-only native OpenFOAM gas adapter**, coupled to the actual CHMT CPU material implementation. It is not the CHMT CUDA backend, and it is not an unmodified `rhoCentralFoam` executable.

## What executes

- The OpenFOAM Foundation 10 `rhoCentralFoam` central-upwind Kurganov flux formula, including its ALE pressure-work correction, and native `fvm::ddt` / `fvc::div` mass, momentum and total-energy equations. These transient systems are diagonal conservative updates, not a PIMPLE pressure solve.
- Two equal-property ideal-gas species, advanced through the same central-upwind flux. This deliberately avoids introducing a second mixture-thermodynamics implementation. The CHMT gas EOS recovers and validates the resulting extensive states.
- Actual native `fvMesh::movePoints`, old-time volume storage and mesh flux. Every microstep compares native volumes and oriented `dt * mesh.phi()` with CHMT polyhedral volumes and exact swept volumes.
- Actual `CpuMaterialDriver::predictWall` and `advanceCandidate`, including native sparse backward-Euler solid conduction, CPU surface constitutive response, material ALE transport and gas-to-material packet consumption.
- Production `evaluateGasWall`, `IntervalHistory`, `compareWallPrograms`, `validateMaterialDonorHistory`, `auditCoupledCandidate` and binary `Checkpoint` read/write. The accumulated physical-target-minus-actual solid-sweep remainder is checked explicitly, retained across accepted windows, and compared bit-for-bit after checkpoint restore.

The verification macro loop is local to this adapter. It is **not** the production `MultirateEvolution` / CUDA `Backend` scheduler. Native fluid fields and trial meshes are recreated from the accepted extensive state for each candidate/replay. This is valid for the Euler one-step scheme used here; it is not evidence for restart of multistep time schemes. The two CHMT checkpoint stage slots both contain the one actually executed Euler step trace. This is an explicitly adapter-specific representation, not evidence of CUDA midpoint-stage checkpoint parity.

## Numerical scope

Gas slab: 0 to 20 mm; solid slab: -4 to 0 mm; common square interface: 10 by 10 mm. The transient meshes have 192 gas and 96 solid tetrahedra. Native `blockMesh` supplies the structured block fixture; an initial-mesh-only conforming tetra decomposition joins each block centre to its triangulated faces. A matched three-plus-one interface diagonal pattern gives full rank for the eight initial surface-sweep constraints. No mesh topology changes occur during a run. Native `checkMesh` must report `Mesh OK` both before integration and at the final moved geometry. Initial gas is 700 K at 100 kPa; initial solid has T=500+40 y/0.01+20 z/0.01 K, creating genuine transverse conduction, interface response and nonuniform recession. Both species have R=287 J/(kg K), cp=1000 J/(kg K) and an energy intercept of 200 kJ/kg. The solid has density 10 kg/m³, cp=1000 J/(kg K), and conductivity 20 W/(m K). Gas conductivity is 2 W/(m K). These are synthetic verification properties, not a calibrated material.

The accepted moving case uses the actual CHMT dry surface Arrhenius reaction: condensed species 0 to gas species 1; A=10, activation energy 15 kJ/mol, no concentration-order factors. Its nonzero mass transfer sets recession through material mass/density; solved gas temperature/pressure and evolving solid temperature feed the next constitutive response. The two-axis thermal variation produces nonuniform two-axis recession and genuinely 3-D conduction/flow response. No bulk gas temperature or wall mass-rate history is prescribed.

An additional negative test applies the same physical model to the original 12×2×2 gas / 4×2×2 solid quad-faced hexahedral mesh. That fixture's harmonic interior motion and exact swept-volume targets conflict with its quad-face planarity constraints and must be rejected transactionally. This is a supported-geometry limitation; it is not a pass of arbitrary moving hex meshes. No planarity tolerance is weakened to admit it. A separately investigated one-axis kinetic proposal was also rejected on the multilayer hexahedra: keeping the front compatible alone does not guarantee planar interior layers under the pinned-boundary harmonic motion policy.


The fixed baseline disables the reaction and motion but retains two-way heat exchange. In the reactive branch, changing interface normals can induce very small normal motion of shared outer-edge vertices. The gas outer boundary is therefore treated as a moving impermeable wall, not silently assumed stationary; its measured normal motion and pressure work are reported. The separate GCL branch is a gas-only uniform state with velocity (3,-2,1) m/s and species fractions (0.8,0.2), on a sinusoidally deformed mesh with fixed end planes and unchanged outer boundary shape; side-boundary vertices may move tangentially. This branch retains uniform through-flow at zero-gradient outer boundaries, unlike the impermeable-wall coupled cases. It checks every conserved gas component, not temperature alone.

Supported claims are limited to these inviscid-with-heat-conduction, conformal, initially planar dry-interface transients. There is no CUDA runtime acceptance, full application acceptance, wet-film coupled CFD, turbulence, particle, shock or general moving-curved-offset validation here.

## Interface ownership and signs

Native flux is positive outward from the gas. CHMT packets are positive into the gas. The **final assembled** native interface flux is replaced by minus the sum of the primary and pore packets divided by dt, for mass, all species, momentum and total energy. It is never added on top of the native interface flux.

CHMT receives `sweptVolume = -gasOutwardSweep`. During recession, gas volume increases and gas pressure work is negative. The packet contains this work exactly once; the native reference solver's mesh pressure-work contribution is replaced at the interface, not added again. Noninterface gas outer walls carry zero relative mass/species flux, outward momentum flux p Sf, and outward energy flux p·meshPhi. Their pressure work is integrated from the actual native swept volume, added to the external boundary-energy budget, and never counted again at the CHMT interface. Solid exterior ALE mass/energy fluxes retain the production material transport budget. Outer impulse is recorded separately.

## Running

Use an actual sourced Foundation OpenFOAM 10 environment:

```
bash applications/CHMT/devtools/of_coupling/check_native_coupling.sh /absolute/evidence/directory
```

No substitute headers, build identity or mock solver is accepted. The script compiles directly against native libraries, creates native meshes with `blockMesh`, and runs all eight cases (seven transient cases and one expected geometric rejection). Logs, `trajectory.csv`, full normalized `state_trajectory.csv`, `summary.json`, checkpoints and final `acceptance.json` are retained in the evidence directory.

The suite evolves each case to the same 0.1 ms physical time. It tests separate gas timestep and material-window refinements using full gas/solid endpoint fields and peak full-field trajectory differences sampled at common macro-output times, maximum per-window and cumulative initial-to-final conservative residuals including all recorded exterior budgets, positivity/CFL, GCL, exactly-once packet budgets, a deliberately rejected real native nonlinear solve, and a binary-checkpoint-restored next window against uninterrupted progression.

`check_results.py` only reads native executable output. It does not generate, integrate, smooth, fit or replace gas/solid simulation results. Geometry errors are normalized by the corresponding cell volume, and inventory errors by actual mass/absolute energy inventory. These normalized audit gates use 512 double-precision machine epsilons. The pass/fail thresholds are executable assertions in that file. Executed/accepted work counters describe the main simulated trajectory and its replays; two additional independent next-window runs validate checkpoint restart and are not included in those counters.

## Cumulative geometry consistency

The native refinement regression exposed accumulation of individually small signed sweep errors in dense material. A per-face carried remainder R=Σ(physical target − measured sweep) is numerical geometry state; the next geometry solve targets the next physical increment plus R. Updating R only from actual measured sweeps bounds the cumulative error rather than allowing a per-window tolerance to accumulate. Physical mass/energy packets and EOS admissibility are unchanged. The native tests verify remainder bounds, dense packing consistency, transactional rejection and checkpoint persistence.
