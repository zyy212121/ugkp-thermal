# GPU-Riemann-UGKP

GPU-Riemann-UGKP is an open-source GPU-resident solver package for
compressible gas-particle flows and multiscale heat-transfer simulations. It
is implemented as a set of OpenFOAM 10 applications with CUDA backends.

The package contains three maintained solvers:

| Solver | Purpose |
| --- | --- |
| `gasUGKP` | Compressible gas flow, UGKP particle transport, statistical particle collisions, and two-way gas-particle momentum and sensible-heat coupling |
| `FSH` | Extends `gasUGKP` with finite-duration particle-wall contact, spreading and retraction, rebound or deposition, particle internal heat conduction, solidification, and wall heat transfer |
| `CHT` | Flow and finite-contact particle-wall heat transfer with two-way solid coupling, transient solid conduction, and particle radiation |

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
    operators/     shared device operations with compile-time field adapters
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
`5e-11 kg` and automatic L1/L2 scheduling with a 1000-step inspection interval.

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

These values are **threads per CUDA block**, not warp counts or the total
number of blocks. A warp contains 32 threads: a value of 64 means two warps
per block. Both dictionary entries default to 128 when omitted.

The CHT 1D cold-wall kernel keeps a separate `coldWallWorkGrid` (total block
count). Its existing compile-time settings are near the top of
`applications/CHT/gpu/GpuResidentStrict.cu`: `coldWallBlockThreads` is 32
threads per block for FP64 and 256 for FP32; `coldWallSmBlocks` is 0 for
FP64 and 48 for FP32. A positive `coldWallSmBlocks` specifies blocks per SM.
When it is zero, `coldWallWorkGrid` inherits `particleWorkGrid`, using the
existing particle-capacity, block-size, and occupancy calculation. These
cold-wall constants are compile-time settings, not `schedulingProperties` keys.

`gpuCsrLevel` accepts `L0`, `L1`, `L2`, and `auto` in this thermal package:

- `L0`: direct atomic reduction without a CP-CST particle-cell directory;
- `L1`: CP-CST directory, warp aggregation, and split pre-transport directory;
- `L2`: L1 plus task segmentation and multi-block reduction for highly occupied cells;
- `auto`: retains the L1 directory and switches heavy-cell segmentation on or off using the existing occupancy criterion.

The thermal parser continues to reject `gpuResearchVariant` and research level
names. `gpuCsrHeavyReductionAutoInterval` is accepted only with `gpuCsrLevel auto`
and must be positive; it defaults to 100. The first particle advance is inspected,
then every configured interval. This is not the Courant update interval.

The existing threshold is `B3 * max(1, ceil(N / (B3 * SM * residentBlocksPerSM)))`,
where N is the current directory population and residency comes from the selected
reduction path. Automatic scheduling enables heavy segmentation only when the
maximum cell population strictly exceeds that threshold. The formula has no
case-fitted constants. Automatic selection does not alter physical operators.
Explicit L2 remains enabled regardless of the criterion; its tile still adapts
to hardware and population without changing the selected level.

## Common numerical operators

The applications instantiate one maintained implementation of each shared
operation through compile-time scalar, field and physical-model adapters.
Particle tracking, collision selection and moments, Gaussian sampling,
weighted sampling correction, task queues, moment recovery and pressure
projection are maintained under `common/`. FSH and CHT additionally share
finite-contact relaxation and capillary/contact finalization.

The adapters preserve native precision, contact-age storage, wall states,
thermal closures and particle payloads. FSH and CHT retain their thermal
application responsibilities; they do not switch into a gas application when
an individual thermal term is inactive. Alternative reduction layouts remain
explicit scheduling policies around the shared operation, without a runtime
solver-name branch in the particle loop.

## Layer names across packages

The fixed thermal levels are `gpuCsrLevel L0/L1/L2`; `auto` selects between
the existing L1 and L2 reduction paths. Their fixed-level operator
bundles correspond to standalone fluid research `L0/S1/S2`, respectively.
The fluid package retains its five research levels `L0/L1/E1/S1/S2`
(T1 is an existing alias of E1). Research L1 is not thermal L1.
No research name or override is accepted by any thermal application, including
thermal gasUGKP. This package-specific configuration header is excluded from
cross-library mirroring; numerical operators remain shared and checked.

## GPU execution and retained optimizations

`gasUGKP`, `FSH`, and `CHT` use the same persistent queue protocol for L2
collision-pool accumulation, particle moments, heavy-cell finalization, and
standalone segmented particle gathering where required. Resident blocks repeatedly claim work until the
queue is exhausted. Each consumer resets its cursor, including when a particle
directory is reused between steps. Worker resources are evaluated for the
actual persistent kernels.

The solvers share task construction, collision-pool operations, particle
transport, moment recovery, pressure operations, and cell-local gathering
while retaining their native physical fields and closures. Gas L2 uses a compact descriptor
directory: each nonempty ordinary cell has one task, and heavy cells have one
task per segment. Empty source cells create no consumer queue entries.
Workers consume these descriptors directly. Filtered heavy cells retain one
owner for stable compaction.

| Optimization | Scope | Work avoided or dependency used |
| --- | --- | --- |
| Deferred granular-temperature read | Gas collision pool | A particle rejected by Poisson sampling does not need its granular temperature; RNG updates and accepted-particle contributions are preserved. |
| Fused directory initialization and task publication | Gas L2 | Existing producer stages initialize counters and publish descriptor totals, removing separate setup passes. Every queue consumer still has its own reset. |
| Compile-time full/base/split directory selection | Gas L2 | A launch's known directory type removes repeated runtime source selection. |
| Pool-target fusion and shared moment recovery | Gas pool targets; Gas/FSH/CHT L2 recovery | Completed cell-local moment sums feed recovery immediately. Single-task and empty cells retain their own recovery path; restart initialization completes all partial reductions before recovery. |
| Reduction-tree pruning | Gas reductions | Shuffle accumulation omits nodes that cannot contribute to the final sum and unused warp-reduction levels. Supported block sizes are 32, 64, 128, and 256. |
| Loop-invariant thermal factors and disabled-exchange specialization | Gas particle relaxation | Cell-invariant factors are reused; particle temperature is not read for a disabled exchange term. |
| Bounded integer particle counts | Gas collision pool | Particle counts use the existing integer-capacity contract rather than floating-point count accumulation. |
| Heavy-cell collision-probability reuse | Gas L2 | A current-step probability is prepared only for multi-segment cells, after pressure preparation and state recovery, and reused by their segments. Ordinary cells compute it in their consuming worker. |
| All-live gather fast path | Cell-local gas/thermal gather | When the existing count proves every source particle survives, gathering skips the filtering scan while copying the full payload. |
| Segmented gather | Gas/FSH/CHT L2 | Independent ranges of heavily occupied all-live cells can be copied by multiple blocks. Filtered cells retain a single owner for stable compaction. |
| Exact-survivor post-transport directory | Thermal gasUGKP/FSH/CHT L1/L2 | The post-transport directory contains exactly the particles kept by compaction, so its offsets identify final positions. Pre-transport and restart directories retain their original rules. |
| Shared moment traversal and reduction | Thermal gasUGKP/FSH/CHT L1/L2 | One common implementation performs survivor validation, eight moment sums and block reduction. Precision, enthalpy and wall-state physics use compile-time adapters. No runtime model dispatch is added. |
| Moment/gather fusion | Thermal gasUGKP/FSH L1/L2 | The same moment traversal copies each surviving particle to its final position. Pressure updates the compact fields before commit; FSH copies every thermal/contact field and publishes each wall-bound index once. |
| Survivor-directory reuse | CHT L1/L2 | The full thermal-payload kernel directly reads the exact source directory after pressure. No separate index-gather pass or redundant source-index writes are needed. |
| Shared particle-field copy | Gas/FSH/CHT | Primary fields and thermal/contact extensions use a shared copy implementation. A compile-time copy-placement policy is independent of the selected layer: Gas/FSH copy during moment traversal; CHT retains a global parallel payload copy after pressure. The field-copy body is maintained once and copies every enabled field exactly once. |
| Shared scheduler, gather, and thermal workers | Gas/FSH/CHT | Common execution code preserves solver-specific particle and wall-contact fields. |
| Shared pressure projection | Gas/FSH/CHT | One cell traversal and mobile-particle update use the limited face flux. Thermal adapters enforce stuck/deposited states and close moments from the constrained particle state. FP32 flat scheduling uses the same pressure-parameter equations. |

These transformations follow data dependencies and supported solver settings;
they do not select policies by case name or by a measured benchmark result.
They preserve the physical models and required particle fields. Floating-point
reduction ordering can affect roundoff and stochastic trajectories. The amount
of elapsed-time improvement depends on the workload and GPU; no fixed speedup
is implied by this list.


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

The CHT radiation model reads an offline alumina Mie table from the path in
the case `constant/radiationProperties`. The large table is not stored in the
Git repository. The thermal `Allrun` runner checks the table before starting
CHT:

- If radiation is disabled, the check is skipped.
- If the referenced file is non-empty, `Allrun` proceeds immediately.
- If the file is missing or empty, `Allrun` asks for interactive
  confirmation before generating it with the default grid. Answer `y` to
  generate the table; any other answer stops before CHT starts.
- Non-interactive jobs must create or copy the table first, then invoke
  `Allrun`.

The supplied radiation cases are `bentSRM_coldWall`,
`MSS7_twoPhase_sparse`, and `MSS7_twoPhase_dense`. Their
`constant/radiationProperties` files refer to
`assets/radiation/alumina_mieTable.dat`.

The generator is:

```text
applications/CHT/thermal/mieTables/make_alumina_mie_table.py
```

For one case, run it from the repository root and write the file to the path
referenced by that case:

```bash
python3 applications/CHT/thermal/mieTables/make_alumina_mie_table.py \
    --out examples/thermal/MSS7_twoPhase_dense/assets/radiation/alumina_mieTable.dat
```

Replace `MSS7_twoPhase_dense` with the selected case name for the other
radiation cases. A single generated table can be copied to the other case
directories when the same diameter and temperature coverage is appropriate;
each case still needs its own file at the path named by `mieTable`.

The default sampling is configurable through command-line options rather than
fixed in the calculation:

| Quantity | Default range/count | Sampling | Options |
| --- | --- | --- | --- |
| Temperature | 300–5000 K, 80 points | linear | `--t-min`, `--t-max`, `--n-t` |
| Particle diameter | 1e-5–4e-4 m, 20 points | logarithmic | `--d-min`, `--d-max`, `--n-d` |
| Scattering angle cosine `mu` | -1–1, 5001 points | linear | `--n-mu` |
| Wavelength | 5e-7–8e-6 m, 120 points | logarithmic | `--lambda-min`, `--lambda-max`, `--n-lambda` |

The diameter and temperature intervals must cover the values reached by the
case. Use `--dry-run` to print the selected grid without writing a file. If
the output already exists, pass `--force` to permit replacement; otherwise the
generator refuses to overwrite it. Generation at the default resolution can
be expensive.

If the referenced file is absent or its ranges do not overlap the particle
diameter/temperature range, CHT stops during radiation preflight before the
first physical time step. Manual preparation or the interactive `Allrun`
confirmation is therefore required for every case with radiation enabled.

## Output and post-processing

OpenFOAM fields and restart data are written to case time directories. GPU
particle restart data preserve parcel identity, random state, particle
thermal history, and finite-contact state.

Runner logs and status summaries are written below the corresponding
`run_logs/` directory. Lightweight reference data, metrics, and figures are
retained under:

```text
examples/consistency/result
examples/performance/results
examples/thermal/results
```

Case-specific Python plotting scripts are stored with persistent case assets,
never below disposable result directories, and can be executed through each
case's `draw.py` after the corresponding simulation has completed.

## Common particle-kernel interfaces

`common/GpuMomentPipeline.cuh` owns the post-transport moment and recovery sequence for thermal gasUGKP/FSH/CHT L1/L2 (fluid research S1/S2). `common/GpuPressurePipeline.cuh` owns the pressure launch protocol; cell traversal, pressure parameters and mobile-particle updates are shared device implementations. Thermal adapters preserve contact state, enthalpy, and constrained moment closure. These are compile-time interfaces, with no runtime application selection.

Within each thermal application, L1/L2 use the same payload placement and physical operations; L2 partitions heavy-cell reductions. Across applications, payload launch sequences need not be identical: CHT keeps its global parallel copy for wide thermal fields, while Gas/FSH fuse copying into moment traversal. Pressure receives the matching original or compact storage view.

The CUDA regression suite checks limited-face local momentum balance, particle/cell moment consistency, conserved totals in a closed two-cell test, survivor indexing, and preservation of enabled thermal/contact fields.

Pressure projection consumes the limited face flux consistently in full and split particle directories. Thermal closure retains constrained particle states and their physical moment accounting.

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

### Particle tracking at geometric boundaries

Particle reflection at `wedge`, `empty`, and symmetry boundaries is non-dissipative. Physical wall restitution and finite-contact thermal laws remain controlled by the wall configuration.

The maximum number of face-walk events per particle and time step is configured in `constant/schedulingProperties`:

```foam
gpuResidentMaxFaceWalkHops 512;
```

The library default is 32. Increase this limit for trajectories that cross or reflect from many faces in one time step, such as passages near a narrow wedge axis. This limit counts all face-walk events, not only physical wall impacts.


公共算子的维护所有权、字段注册、构建检查及 CUDA 调度契约见 [双库公共算子维护](docs/OPERATOR_MAINTENANCE_ZH.md)。


### 公共算子维护入口（2026-10-03 收口）

gas、FSH、CHT 的公共算子核及主机流程由 `common/` 维护。公共文件上游为 `ugkp-thermal/common`；gas 应用入口上游为独立 gas 库。修改公共实现后运行 `python3 tools/managed_mirrors.py --sync`，按登记归属同步双库；构建前检查会拒绝镜像漂移。无需分别移植三套核心实现。各应用的能力/精度适配仍需各自验证。当前生产双库冻结期间不要运行跨库 `--sync`；后续镜像同步须单独审核，公共调度配置头不在镜像范围内。工程结论与十对 k1/k2 结果见 [本轮结果](docs/OPERATOR_CONSOLIDATION_R2_RESULTS_ZH.md)，维护契约见 [维护说明](docs/OPERATOR_MAINTENANCE_ZH.md)。

本次自动调度恢复、最后三项代码清理及验证结果见 [2026-10-04 收尾验证](docs/development/auto-cleanup-20261004/README.md)。

`gasUGKP auto` 的任务计数与切档后任务准备已补充修复；实际碰撞池消费回归及完整测试集限制见 [gas auto 正确性修复](docs/development/gas-auto-correctness-20261004/README.md)。

三应用的 full/split 目录准备与 auto 调用现统一为公共主机流程，gas 无注入正确选择 baseOnly 并复用任务。生产调用链、守恒矩对照和缓存复用验证见 [目录主机流程统一](docs/development/directory-host-unification-20261004/README.md)。
