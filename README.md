# GPU-Riemann-UGKP

GPU-Riemann-UGKP is an open-source GPU-resident solver package for
compressible gas-particle flows and multiscale heat-transfer simulations. It
is implemented as a set of OpenFOAM 10 applications with CUDA backends.

The package contains three maintained solvers:

| Solver | Purpose |
| --- | --- |
| `gasUGKP` | Compressible gas flow, UGKP particle transport, statistical particle collisions, and two-way gas-particle momentum and sensible-heat coupling |
| `FSH` | Extends `gasUGKP` with finite-duration particle-wall contact, spreading and retraction, rebound or deposition, particle internal heat conduction, solidification, and wall heat transfer |
| `CHT` | Extends the common gas-particle framework with transient solid conduction, particle radiation, and conjugate fluid-particle-solid heat transfer |

The three solvers share the numerical and physical implementations under
`common/`.

## Requirements

The released version has been built and tested with:

- Ubuntu 22.04 or Ubuntu 22.04 under WSL2;
- OpenFOAM 10;
- GCC 11.4;
- CUDA 13.1;
- an NVIDIA CUDA-capable GPU;
- Python 3 with NumPy, SciPy, and Matplotlib for preparation and
  post-processing scripts.

The default CUDA target is `sm_89`. For another NVIDIA architecture, set
`UGKWP_CUDA_ARCH` before compilation, for example:

```bash
export UGKWP_CUDA_ARCH=sm_80
```

## Obtaining and building the software

```bash
git clone https://github.com/zyy212121/ugkp-thermal.git
cd ugkp-thermal

source /opt/openfoam10/etc/bashrc
export CUDA_HOME=/usr/local/cuda

./Allwmake
```

`Allwmake` checks the OpenFOAM environment and CUDA compiler and then builds
all three solvers. The generated executables are installed in
`$FOAM_USER_APPBIN`:

```text
gasUGKP
gasUGKPCudaBackend
FSH
FSHCudaBackend
CHT
```

A successful build ends with `All solvers built successfully.` and prints a
source hash and the hashes of the generated executables.

## Repository structure

```text
applications/
    gasUGKP/       base gas-particle solver
    FSH/           finite-contact particle heat-transfer solver
    CHT/           conjugate heat-transfer and radiation solver

common/
    gasNumerics/   shared gas-phase fluxes and reconstruction methods
    gpu/           shared GPU scheduling and coupling utilities
    wall/          shared finite-contact and particle thermal models

examples/
    consistency/   numerical and gas-particle consistency tests
    performance/   single- and two-phase nozzle performance cases
    thermal/       particle-wall and conjugate heat-transfer cases
```

All cases follow the standard OpenFOAM directory structure. Initial fields
are stored in `0/` or in a retained non-zero starting-time directory when a
prepared checkpoint is required. All dimensional quantities use SI units.

## Running the supplied cases

Load the OpenFOAM environment before invoking a runner:

```bash
source /opt/openfoam10/etc/bashrc
```

Use `--list` to show the valid case names accepted by a runner. With no case
name, a runner executes all cases in that group sequentially.

### Consistency cases

```bash
cd examples/consistency

./Allrun --list
./Allrun sodShockTube
./Allrun planarCouette
./Allrun dustyBox
./Allrun dustyWave
./Allrun windSandShockTube
```

`dustyWave` and `windSandShockTube` use validation-specific solver adapters.
Their local `Allrun` scripts stage the current production source, compile the
private validation executable, run the case, and remove the temporary build
products. The production solver source is not modified. Additional details
are provided in the README inside each of these case directories.

### Single-phase nozzle cases

```bash
cd examples/performance

./Allrun01 --list
./Allrun01 slau2_2
```

Running `./Allrun01` without an argument evaluates all supplied gas-flux
configurations:

```text
tadmor
kurganov
hlle
hllc
roe
hllem
hllc_adc
slau2
slau2_2
```

### Two-phase nozzle cases

```bash
cd examples/performance

./Allrun02 --list
./Allrun02 sparse
./Allrun02 dense
./Allrun02 denseL2
```

The cases use the following particle scheduling configurations:

| Case | GPU particle-cell level |
| --- | --- |
| `sparse` | L0 |
| `dense` | L1 |
| `denseL2` | L2 |

The performance runners refuse to start when another GPU compute process is
active, preventing overlapping benchmark workloads.

### Thermal cases

```bash
cd examples/thermal

./Allrun --list
./Allrun singleAluminaDrop/coldWall
./Allrun bentSRM_coldWall
./Allrun MSS7_laminar
./Allrun MSS7_turbulent_wallModel
./Allrun MSS7_twoPhase_sparse
./Allrun MSS7_twoPhase_dense
```

Some thermal cases are computationally expensive and may require several
hours or longer depending on the GPU.

`MSS7_twoPhase_sparse` preserves the development-case parcel mass of
`1e-9 kg` and uses L0 scheduling. `MSS7_twoPhase_dense` uses a parcel mass of
`5e-11 kg` and L2 automatic scheduling with a 1000-step inspection interval.

## Cleaning generated case data

The cleaning scripts remove generated time directories, run logs, meshes,
temporary executables, and private build products while retaining the
official initial condition and case inputs.

```bash
cd examples/consistency
./Allclean
./Allclean dustyWave

cd ../performance
./Allclean01
./Allclean01 slau2_2
./Allclean02
./Allclean02 denseL2

cd ../thermal
./Allclean
./Allclean bentSRM_coldWall
./Allclean MSS7_twoPhase_sparse
./Allclean MSS7_twoPhase_dense
```

## Case configuration

The principal input files are:

| File | Contents |
| --- | --- |
| `system/controlDict` | Solver name, start and end times, time-step limits, Courant control, and output interval |
| `system/fvSchemes` | Gas flux, time integration, spatial reconstruction, gradients, interpolation, Laplacian, and surface-normal gradient schemes |
| `system/fvSolution` | Positivity limits, diffusion limits, robust flux fallback, and field-solver settings |
| `constant/fluidProperties` | Equation of state, molecular weight, heat capacity, viscosity, Prandtl number, and laminar, LES, or RAS model |
| `constant/particleProperties` | Particle material properties, size distribution, parcel mass, injection tables, drag, gas-particle heat transfer, collisions, and wall interaction |
| `constant/schedulingProperties` | GPU particle capacity, Courant update interval, CUDA block sizes, and particle-cell scheduling level |
| `constant/radiationProperties` | Radiation activation, angular discretization, particle optical table, and radiation coupling interval |
| `constant/solidRegionProperties` | Solid density, heat capacity, thermal conductivity, and CHT region configuration |

### Particle distribution and injection

The particle-size distribution is controlled by:

- `dS`: characteristic particle diameter;
- `dMin`: lower diameter limit;
- `dMax`: upper diameter limit;
- `dSigma`: logarithmic distribution-width parameter.

`parcelMass` or `injectionParcelMass` specifies the physical particle mass
represented by one computational parcel.

Time-dependent inlet pressure and particle volume fraction are specified by
`gpuResidentPressureTable` and `gpuResidentVolumeFractionTable`. The GPU uses
piecewise-linear interpolation between table entries.

`gpuResidentRandomSeed` controls particle injection and stochastic sampling.
It should remain unchanged when reproducing archived results.

Available particle drag models are:

```text
none
SchillerNaumann
GidaspowErgunWenYu
```

Available gas-particle sensible-heat models are:

```text
none
RanzMarshall
```

## Gas-phase numerical options

The gas flux is selected with the top-level `fluxScheme` entry in
`system/fvSchemes`. Supported schemes are:

```text
Tadmor
Kurganov
HLLE
HLLC
Roe
HLLEM
HLLC-ADC
SLAU2
SLAU2.2
```

For contact-resolving or low-dissipation schemes,
`fvSolution/UGKP/robustFallback` should normally remain `true`. Invalid
intermediate states are then replaced locally by a more robust HLLE or
Rusanov flux.

Available explicit time integrators are:

```text
Euler
SSPRK2
SSPRK3
```

The reconstruction is determined by the entries in `divSchemes`:

- `Gauss upwind` for all transported terms gives first-order reconstruction;
- momentum upwind with `Gauss limitedLinear 1` for the energy terms gives
  limited energy reconstruction;
- `Gauss MUSCL` enables MUSCL reconstruction.

Available MUSCL limiters are `none`, `barthJespersen`, and
`venkatakrishnan`.

The current GPU implementation expects:

```text
gradSchemes/default          Gauss linear
interpolationSchemes/default linear
snGradSchemes/default        corrected
laplacianSchemes/default     Gauss linear corrected
```

## GPU scheduling

GPU scheduling is controlled by `constant/schedulingProperties`.

| Entry | Meaning |
| --- | --- |
| `gpuResidentPureGasOnly` | Enables the pure-gas execution path when no particles are present |
| `gpuResidentParticleCapacity` | Maximum number of resident computational parcels |
| `gpuResidentCourantUpdateInterval` | Number of time steps between Courant-number evaluations |
| `gpuParticleBlockThreads` | CUDA block size for particle-transport kernels |
| `gpuReductionBlockThreads` | CUDA block size for particle-cell reduction kernels |
| `gpuCsrLevel` | Particle-cell data-path level |

The block sizes can be 32, 64, 128, or 256.

`gpuCsrLevel` has four valid settings:

- `L0`: direct atomic reduction without a CP-CST particle-cell directory;
- `L1`: CP-CST directory, warp aggregation, and split pre-transport
  directory;
- `L2`: L1 plus task segmentation and multi-block reduction for highly
  occupied cells;
- `auto`: retains the L1 directory and periodically determines whether L2
  heavy-cell segmentation should be activated.

## Particle-wall models

Wall patches not listed in `gpuResidentStuckWallPatches` use the ordinary
instantaneous reflection model. Their restitution coefficients are specified
in `particleWallCoeffs`.

Persistent finite-contact processing is enabled only for wall patches listed
in `gpuResidentStuckWallPatches`. The model is selected with
`wallInteractionModel` in `gpuResidentStuckModel`, or overridden for an
individual patch in `wallInteractionModels`.

Available finite-contact models are:

- `reboundContact`: finite-duration spreading and retraction followed by
  rebound, without long-term deposition;
- `coldWall1D`: Sommerfeld-based rebound/deposition selection with
  one-dimensional particle internal enthalpy conduction, latent heat,
  solidification, and contact-line pinning;
- `coldWall2D`: axisymmetric radial-normal particle internal conduction. This
  model is retained as an experimental extension; the published thermal
  validation uses `coldWall1D`.

Important finite-contact parameters are:

| Entry | Meaning |
| --- | --- |
| `sommerfeldThreshold` | Threshold separating rebound and deposition tendencies |
| `maximumCoverage` | Maximum fractional wall-face area covered by contacting particles |
| `reflectionHeatTransferEfficiency` | Effective heat-transfer-area efficiency during finite spreading and retraction |
| `depositionHeatTransferEfficiency` | Effective heat-transfer-area efficiency during long-term deposition |
| `contactAngleDegree` | Particle-wall contact angle used by spreading and capillary adhesion |
| `adhesionEnergyScale` | Multiplier for the capillary detachment-energy barrier |
| `interfaceThermalResistance` | Additional particle-wall interfacial resistance in m2 K W-1 |
| `wallTransientResistance` | Enables wall-side transient thermal resistance based on contact age |
| `meltingTemperature` | Particle melting temperature |
| `mushyRange` | Temperature width of the mushy region |
| `latentHeat` | Particle latent heat |
| `pinningThicknessFraction` | Connected-solid thickness required for contact-line pinning |
| `solidificationIterations` | Nonlinear particle enthalpy iterations per time step |

In `FSH`, wall density, heat capacity, and thermal conductivity are specified
in the finite-contact dictionary. In `CHT`, the contact model reads the wall
material from `constant/solidRegionProperties`, ensuring consistency with the
solid conduction equation.

## Particle-radiation table

The large alumina Mie table used by `bentSRM_coldWall`,
`MSS7_twoPhase_sparse`, and `MSS7_twoPhase_dense` is generated locally rather
than stored in the Git repository. The thermal `Allrun` entry generates one
cached table and links it into each selected radiation case automatically.
It can also be generated manually for an individual case:

```bash
python3 applications/CHT/thermal/mieTables/make_alumina_mie_table.py \
    --out examples/thermal/bentSRM_coldWall/assets/radiation/alumina_mieTable.dat \
    --force
```

The generated file is referenced by the `mieTable` entry in the case
`constant/radiationProperties`.

## Output and post-processing

OpenFOAM fields and restart data are written to case time directories. GPU
particle restart data preserve parcel identity, random state, particle
thermal history, and finite-contact state.

Runner logs and status summaries are written below the corresponding
`run_logs/` directory. Lightweight reference data, metrics, figures, and
plotting scripts are retained under:

```text
examples/consistency/result
examples/performance/results
examples/thermal/results
```

Case-specific Python plotting scripts can be executed after the corresponding
simulation has completed.

## Reproducibility notes

- Use the Git commit or release tag cited in the associated manuscript.
- Keep the supplied random seeds unchanged for stochastic particle cases.
- Do not run two GPU cases concurrently on the same GPU.
- Numerical results are not tied to the NVIDIA GPU model used for the
  published timing measurements, but wall-clock performance is hardware
  dependent.
- Large generated time directories are not stored in the repository. The
  supplied initial conditions, preparation scripts, lightweight comparison
  data, and post-processing scripts are provided to recreate them.
- Performance comparisons should report the GPU, CPU, CUDA version,
  OpenFOAM version, compiler, precision, and selected scheduling level.

## Citation

If this software contributes to published work, please cite:

Shashi Liu, Changmeng Liu, and Xiao Hou, "GPU-Riemann-UGKP: Open-Source GPU
Software for Multiscale Heat Transfer in Gas-Particle Flows," Computer
Physics Communications, manuscript COMPHY-D-26-00986.

## License

Copyright (C) 2026 Shashi Liu.

This software is distributed under the GNU General Public License, version 3
or any later version (`GPL-3.0-or-later`). See [LICENSE](LICENSE) and
[NOTICE](NOTICE).
