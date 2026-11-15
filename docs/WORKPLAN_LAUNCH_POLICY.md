# Startup launch policy

This document describes the new implementation, not a completed numerical acceptance result.

## Configurable launch families

The startup reader in common/GpuSchedulingConfiguration.H publishes the following constant/schedulingProperties keys to common/gpu/WorkPlan.H. Every omitted key defaults to 128. A supplied value selects the family width; no startup timing competition is performed.

| Key | Work |
| --- | --- |
| gpuCellBlockThreads | Gas cell work, cell-indexed CHMT and geometry work |
| gpuFaceBlockThreads | Gas face work, flat boundary and surface transfer, flat point/face geometry work |
| gpuParticleBlockThreads | Particle-indexed work |
| gpuReductionBlockThreads | Particle moments and general block reductions; geometry wall-distance and cooperative motion reductions |
| gpuSparseBlockThreads | Public sparse solve and cooperative geometric QR work |
| gpuRadiationBlockThreads | DOM, optical properties and radiation field work |
| gpuThermalBlockThreads | Shared conduction and thermal field work |
| gpuMaterialBlockThreads | CHMT material-indexed work and its applicable reductions |
| gpuColdWallBlockThreads | 1D cold-wall particle work; eight lanes per particle remain unchanged |

These values are read at startup before resident owners and CUDA graphs are created. They are not per-step tuning controls. Captured launch shapes and scratch capacity must use the same selected policy. L0/L1/L2/auto describe task organization, not a second definition of block width.

## Algorithmic exceptions

- Scalar control and state publication kernels launched with one thread remain one thread. Increasing their launch size would duplicate scalar updates.
- The eight-lane wall boundary-layer tile retains its grouping but uses gpuFaceBlockThreads for the enclosing CTA. Workers per CTA are blockDim.x/8 and the grid uses that same divisor. There is no CTA-local shared scratch or 32-thread launch bound in this tile path.
- The full-warp wall boundary-layer team and conservative wall reconstruction remain 32 threads per CTA: CudaWallTeam uses threadIdx.x as its rank and assigns one complete workspace to each CTA. These are distinct from the independently packed eight-lane tile path.
- CHMT surface teams retain the existing Team::width specializations (32, 64 or 96) and their matching launch bounds, shared storage and cooperative surface algorithm. These are not independently overridden by an ordinary flat-work block key.
- Optional sparse Block8 factorization and application use `gpuSparseBlockThreads` (default 128) for their enclosing CTA. The eight-row factor and eight-entry triangular-solve scratch are algorithm sizes, not fixed launch widths. One CTA still owns one matrix block; lanes beyond the first eight only participate in uniform barriers. Jacobi remains the default, and the existing small-system path is unchanged. Idle-lane cost must be included in actual-matrix performance measurements.
- CUDA/CUB library-owned launches remain controlled by the library rather than being rewritten as application kernels.

Fixed scratch storage alone is not an algorithmic exception. In particular, the former 128-thread wall-distance reduction and 256-thread cooperative geometry motion reduction now use selected widths, dynamic shared storage and matching occupancy calculations. Their reductions include odd active counts. Geometric QR also uses the selected sparse width; it is not fixed merely because its former default was 128.

## 1D cold wall

Eight threads cooperate on each particle's eight-node solve. At the default block width of 128 there are sixteen independent particle groups per block. The block setting does not change the one-particle discretization into 128 nodes and does not turn this into a heavy-cell L2 task.

## Engineering checks and remaining acceptance

The removed GpuToolB1/GpuToolB1Launch timer interfaces are replaced by GpuWorkPlanPublish.cuh. The managed-mirror inventory and active timer-related regression assertions now reference startup publication instead of timing physical kernels.

PolyhedralGeometry.cu has compiled successfully with CUDA sm_89, minimal fast compilation and relaxed constexpr support. This is a compilation check only. Full maintained-application FP32/FP64 builds and the final original-baseline restart comparison remain required before acceptance.

The launch audit must inspect instantiated cooperative widths and CUDA graph capture paths as well as literal triple-chevron launches. A grep without fixed numeric launches is not proof that all manual overrides are correct.


## Independent material surface workers
The material family keeps the exact configured CTA width (default 128).
For a small pool of expensive independent scalar workers, the active worker
prefix per CTA is ceil(workers / residentCTAs), capped by the configured width.
residentCTAs is SM count times static occupancy at that width. A worker keeps
its original face stride and private workspace index. Inactive lanes return
before the scalar solve. Large pools use dense CTA prefixes. This does not
change fixed cooperative wall-team algorithms and does not run timed tuning.
