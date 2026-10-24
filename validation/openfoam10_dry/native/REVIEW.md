# PR24 mathematical and native execution review (2026-10-25)

Reviewed against d867975, stacked on PR23 46f1c85. Extensive condensed/pore
inventories, formation-inclusive internal energy, corrected species diffusion,
Darcy enthalpy transport, separate geometric internal-energy sweep, area
conversion and accepted gas packet ownership were traced through the production
CPU adapters and native gas execution. No liquid-film extension is claimed.

## Corrections

* **Explicit pore diffusion stability:** current mass/outflow alone permits an
  arbitrarily large step near uniform composition. A two-cell perturbation
  reproduces the defect on the original source. The corrected step sums the
  frozen diffusion operator's largest diagonal loss coefficient (including its
  upwind mass-correction channel), divided by total pore mass. This is combined
  with existing Darcy/ALE stiffness in the material local controller. The gas
  timestep and shared gas equations are unchanged. With rho=2, D=1e-4,
  A=.01, d=.1 and pore mass=.001, CFL=.4 now gives 20 s and the two-cell
  perturbation amplification is .2; the original donor-only step amplifies it.
* Invalid negative/nonfinite pore diffusivities fail at material transport
  consumption instead of producing antidiffusion.
* Record poreEffectiveDiffusivity in checkpoint model metadata and advance the
  layout declaration to schema 8. Active controls still come from the caller;
  the pre-existing restart admission policy is unchanged.
* Remove retired CpuFilmDriver linkage from the native Allwmake entrypoint.
* Add optional CHMT_CUDA_STACK_BYTES, matching gasUGKP's existing runtime
  facility. Fast compilation outlined thermo calls and Compute Sanitizer
  identified stack overflow at the default stack. 65536 bytes passed the native
  cases and sanitizer. The default for normal builds is unchanged.

## Reproduced validation

* Seven dry core/adapter tests with UBSan and warnings-as-errors passed, including
  Ns=3 disparate diffusivities/zero-support species and the new weak-perturbation
  regression. The new regression failed before the stability correction.
* Actual OpenFOAM10 fvMesh/fvScalarMatrix suite passed: 1D/3D heat and species
  diffusion, Darcy pressure, reaction kinetics, unequal diffusivities, independent
  SciPy coupled reference. Production CPU adapter conduction error 1.14e-13 K;
  accepted reacting packet mass residual -8.34e-14 kg and energy residual
  5.96e-8 J. Checkpoint behavioral regression passed.
* Full native CHMT CUDA compilation, host linkage and RTX 4060 execution passed
  thermal, single-gas, multicomponent diffusion, porous bulk pyrolysis, and
  moving surface-recession cases. Whole-domain mass residual rounded to zero;
  relative energy residual <=3.96e-16 in these runs. Independent accepted
  exchange-energy checker also passed thermal/single/diffusion.
* Native dry pyrolysis mass error against 500 exp(-2t), at t=.001 s, decreased
  from 4.99334e-4 to 2.49584e-4 kg when the coupling/material step was halved:
  first-order temporal convergence, consistent with the explicit local map.
* Native dry case under Compute Sanitizer memcheck: zero errors.

Hardware/toolchain: WSL Ubuntu22.04, Foundation OpenFOAM10, GCC11, CUDA13.1,
sm_89 RTX4060. CUDA flags: -std=c++17 -O2 --Ofast-compile=max
-Xcompiler=-fno-tree-fre; host -O2, Ns=2. These are native cubins, not PTX-only
evidence. Fast compilation is recorded explicitly; this is not a claim that
the default fully optimizing release build was benchmarked.

## Reproduction

Source OpenFOAM10, set CUDA_HOME and CHMT_CUDA_ARCH=sm_89, and build via
applications/CHMT/Allwmake into a private CHMT_BUILD/FOAM_USER_APPBIN. For the
recorded fast-compile flags, set NVCC to a wrapper that execs nvcc with
--Ofast-compile=max -Xcompiler=-fno-tree-fre before forwarding its arguments,
and set CHMT_CUDA_STACK_BYTES=65536 when running. Use:

```
python3 -m pytest --noconftest -q validation/tests/test_dry_material_{bulk,adapter,surface}.py validation/tests/test_dry_surface_adapter.py
bash validation/openfoam10_dry/run_all.sh
bash validation/openfoam10_dry/adapter/run_adapter.sh
python3 validation/openfoam10_dry/native/run_native.py --binary "$CHMT_BUILD/CHMT" --cases-root "$UGKP_CASES_ROOT" --output "$WORK/native"
```

The cases checkout supplies the existing generators and independent checker;
check_coupled.py only declares the new schema hash, retaining its conservation,
exchange, checksum and refinement assertions. The local checkout's generator
hashes and native binary hash are recorded in evidence.json. Every run requires
a fresh output directory. Use --scenarios dry --refine and --scenarios dry
--memcheck for the temporal and sanitizer checks.

## Scope and retained limits

The model is LTE dry material, two production condensed pools, mass-averaged
Fick diffusion and Darcy transport, with one effective exposed phase or congruent
recession. It is not an experimental material calibration or arbitrary selective
multisolid front model. Coupling remains one-pass first order: existing normalized
wall-waveform accuracy warnings occur on these finite windows, are retained in
logs, and are not reported as nonlinear convergence. Generic all-host legacy
check_host.sh stops at its pre-existing C++14 versus C++17 warning-as-error;
the explicit dry, OF10, checkpoint and native checks above are the evidence.

## Native split/restart check

A thermal run split at 0.0005 s and resumed to 0.001 s has an identical
122696-byte physical checkpoint payload and byte-identical terminal accepted,
gas, material and exchange CSVs compared with the uninterrupted run. The state
SHA-256 is 4189d492b16465982595bf3197e84d8be89e2480ef1cd1ec12c90a78e369dbfa.

The external contact-conductance oracle assumes a 2x2x2 mesh. Larger-grid timing
runs use the independent extensive-inventory check and before/after physical
checkpoint comparison; they do not reuse that fixed-distance analytic oracle.
