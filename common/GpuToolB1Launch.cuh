#pragma once

// Common diagnostic launch bundles only. Requires GasHostPolicy::Time,
// DeviceState, gas kernels and setLastError. Tuning/occupancy/event handling
// stays with the consumer because those failure protocols differ.
static constexpr int toolB1WarmupRuns = 1;
static constexpr int toolB1MeasuredRuns = 5;

int launchToolB1CellBundle(DeviceState* s, const int block)
{
    const int grid = (s->nCells + block - 1)/block;
    recoverGasPrimitivesKernel<<<grid, block>>>(s->deviceState);
    if (s->hostTurbulenceModel == 3)
    {
        recoverSstPrimitivesKernel<<<grid, block>>>(s->deviceState);
        applySstWallFunctionStateKernel<<<grid, block>>>(s->deviceState);
    }
    if (s->hostGasFluxScheme == 7)
    {
        computeGasHllcAdcSensorKernel<<<grid, block>>>(s->deviceState);
    }
    computeGasPrimitiveGradientsKernel<<<grid, block>>>(s->deviceState);
    computeGasGradientLimiterKernel<<<grid, block>>>(s->deviceState);
    computeGasEddyViscosityKernel<<<grid, block>>>(s->deviceState);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("ToolB1 gas-cell bundle launch", err);
        return 1;
    }
    return 0;
}

int launchToolB1FaceBundle
(
    DeviceState* s,
    const int block,
    const GasHostPolicy::Time dt,
    const GasHostPolicy::Time simulationTime
)
{
    const int grid = (s->nFaces + block - 1)/block;
    if (grid <= 0)
    {
        return 0;
    }
    updateLegacyGasBoundaryMirrorKernel<<<grid, block>>>
    (
        s->deviceState,
        simulationTime
    );
    updateRiemannBoundaryMirrorKernel<<<grid, block>>>(s->deviceState);
    if (s->hostTurbulenceModel == 0)
    {
        computeGasInternalFaceFluxKernel<false><<<grid, block>>>
        (
            s->deviceState,
            dt
        );
    }
    else
    {
        computeGasInternalFaceFluxKernel<true><<<grid, block>>>
        (
            s->deviceState,
            dt
        );
    }
    if (s->hasPeriodicFaces != 0)
    {
        enforcePeriodicGasFluxAntisymmetryKernel<<<grid, block>>>
        (
            s->deviceState
        );
    }
    if (s->hostTurbulenceModel == 3)
    {
        const int cellGrid = (s->nCells + block - 1)/block;
        computeSstGradientsKernel<<<cellGrid, block>>>(s->deviceState);
        computeSstFaceFluxKernel<<<grid, block>>>(s->deviceState);
        if (s->hasPeriodicFaces != 0)
        {
            enforcePeriodicSstFluxAntisymmetryKernel<<<grid, block>>>
            (
                s->deviceState
            );
        }
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("ToolB1 gas-face bundle launch", err);
        return 1;
    }
    return 0;
}
