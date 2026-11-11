# Checkpoint 2: FP32 preconditioning with FP64 physical equations

Implementation scope only. Numerical tests and timing are deferred to the joint checkpoint-2 batch. This change is for the isolated FP32 source overlay, not the preserved FP64 tree.

## Production chain and retained precision

- `FoamImplicitDiffusion` / `FoamSparseSolve` prepare the original finite-volume matrix and call `SparseConjugateGradient` for CHT and CHMT material conduction.
- `ImplicitDiffusion` owns FP64 conductivity, heat capacity, matrix factors, temperatures, gradients, boundary fluxes and residuals. Its physical coefficient assembly and interpolation are unchanged.
- `SolidThermalProperties`, `SolidThermalIteration`, CHMT `MaterialBackend` and `CpuMaterialDriver` retain FP64 authoritative material quantities and the existing scaled thermal energy accounting. CHMT `core/Types.H` continues to define `Real` as `double`.
- Gas species diffusion is a separate templated face-flux operator (`SpeciesDiffusion.H`, used by `computeRiemannGasFaceFluxDevice.cuh`), not this sparse solve. Its carrier closure and enthalpy flux are unchanged here. No claim of migrating its authoritative heat/species audit is made.

## Implemented low-precision portion

- A prepared FP32 diagonal copy and FP32 `z` preconditioner workspace are private to the sparse owner.
- Normal rows evaluate `float(r64) / diagonal32`; the quotient is stored as FP32 and converted to FP64 before outer PCG arithmetic.
- Outer `x`, recursive `r`, `p`, `Ap`, physical CSR/LDU coefficients, affine jumps, normalization and all dot/norm reductions remain FP64. No pure-FP32 solve is introduced.
- The diagonal copy is rebuilt from the current physical matrix for each solve, including replayed resident coefficient updates.

## Range continuation and meaning of the private NaN tag

A normal finite FP32 input/result uses the compact path. If a row cannot be represented normally, its `z32` slot receives a private NaN tag, not a physical result. Every consumer, including direction updates and rho reductions, decodes that tag by evaluating the original `r64 / diagonal64` directly into an FP64 register. The FP64 value is never narrowed back to the tagged slot. There is no second double-z array, no new rejection, no rollback and no changed tolerance. Original genuinely non-finite arithmetic remains genuinely non-finite; decoding does not repair or hide it.

`ConjugateGradientReport::preconditionerFallbackApplications` counts row applications that took this continuation. It counts applications, not distinct rows; distributed reports contain the local rank count. The initial FP32 cast and private tag are never exposed as physical output.

## Real residual and convergence

The recursive PCG vector is retained for the recurrence, but it is no longer used as the published or stopping residual. When the recursive residual claims convergence or reaches an existing iteration-limit/singularity report point, the solver re-evaluates `b - (A64*x64 + affine64)` in original CSR order, including native interface contributions. The original finite-volume norm factor, absolute/relative tolerances, minimum/maximum iterations and singular policy are retained. A failed candidate is not accepted: replace the recursive residual with the true residual, refresh the preconditioned state and restart the next direction with beta=0 while retaining the accumulated iteration count. `initialResidual`, `finalResidual` and `converged` therefore use physical FP64 residuals. The report counts `physicalResidualEvaluations` (including the initial evaluation) and `residualRestarts`. Existing explicit `residualResidentLdu` and thermal energy audit paths are unchanged.

## Storage and performance limits

For a normal preconditioned row, diagonal input + residual input + z output is nominally 4 + 8 + 4 = 16 bytes instead of 8 + 8 + 8 = 24 bytes, excluding topology and scalar/reduction traffic. Direction reads z at 4 rather than 8 bytes. Division is FP32. Exceptional rows additionally read the FP64 diagonal and use FP64 division.

The extra diagonal32 copy (4 bytes/row) balances the saved z storage (4 bytes/row); total local preconditioner workspace is not halved. No extra FP64 true-residual vector is retained: verification replaces the recursive residual at the check point. Verification adds a physical FP64 matvec only at candidate/final checks and, for distributed interfaces, a matching interface exchange at those points. In the multi-block resident path, two extra graph kernel nodes are present per stage; both return immediately when no check is required. Frequent false candidates, launch overhead or exceptional-row decoding may outweigh the preconditioner savings. No whole-solver speedup or memory reduction is claimed without the joint batch measurements. The main sparse matvec remains FP64.

No capacity policy, direct-device entry point, graph batch size or scheduling optimization is included. The existing four-iteration graph batching and single-block path remain. Interface rho uses the same precondition-and-dot helper so that private tags are decoded identically.

## Deferred fixtures

`validation/tests/fp32_sparse_preconditioning.cpp` covers nonuniform SPD systems, strong contrast, both sides of the single-block threshold, callback/interface paths, independently recomputed true residual, global interior-flux cancellation, FP32 range continuation and original invalid arithmetic semantics. Existing sparse/resident/CHMT thermal fixtures must join the checkpoint-2 batch. In particular, `resident_sparse_publication.cu` assumes exact two-step finite termination on a 2x2 problem; inexact FP32 preconditioning may require extra iterations while preserving its 1e-12 residual and 1e-11 solution targets. That iteration-count expectation must be reviewed during the joint batch, not used to loosen physical accuracy targets. The new fixture is not run during implementation.
