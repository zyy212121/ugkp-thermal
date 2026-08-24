#pragma once

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
    if (s->hostTurbulenceModel == 3)
    {
        computeSstGradientsKernel<<<grid, block>>>(s->deviceState);
    }
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
    const double dt,
    const double simulationTime
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

template<class Launch>
int measureToolB1Candidate
(
    Launch launch,
    cudaEvent_t start,
    cudaEvent_t stop,
    float& median,
    float& measuredTotal
)
{
    for (int run = 0; run < toolB1WarmupRuns; ++run)
    {
        if (launch() != 0)
        {
            return 1;
        }
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        setLastError("ToolB1 warmup", err);
        return 1;
    }
    std::vector<float> samples;
    samples.reserve(toolB1MeasuredRuns);
    for (int run = 0; run < toolB1MeasuredRuns; ++run)
    {
        err = cudaEventRecord(start);
        if (err == cudaSuccess && launch() != 0)
        {
            return 1;
        }
        if (err == cudaSuccess)
        {
            err = cudaEventRecord(stop);
        }
        if (err == cudaSuccess)
        {
            err = cudaEventSynchronize(stop);
        }
        float elapsed = 0.0f;
        if (err == cudaSuccess)
        {
            err = cudaEventElapsedTime(&elapsed, start, stop);
        }
        if (err != cudaSuccess)
        {
            setLastError("ToolB1 CUDA event measurement", err);
            return 1;
        }
        samples.push_back(elapsed);
        measuredTotal += elapsed;
    }
    const size_t middle = samples.size()/2;
    std::nth_element
    (
        samples.begin(),
        samples.begin() + middle,
        samples.end()
    );
    median = samples[middle];
    return 0;
}

int tuneFixedWorkBlockThreads
(
    DeviceState* s,
    const double dt,
    const double simulationTime
)
{
    if (s->fixedWorkBlockTuned != 0)
    {
        return 0;
    }
    const int candidates[] = {32, 64, 96, 128, 160, 192, 224, 256};
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaError_t err = cudaEventCreate(&start);
    if (err == cudaSuccess)
    {
        err = cudaEventCreate(&stop);
    }
    if (err != cudaSuccess)
    {
        if (start != nullptr)
        {
            cudaEventDestroy(start);
        }
        setLastError("ToolB1 CUDA event creation", err);
        return 1;
    }

    int bestCellBlock = 0;
    int bestFaceBlock = 0;
    float bestCellMs = 0.0f;
    float bestFaceMs = 0.0f;
    float measuredTotal = 0.0f;
    for (const int block : candidates)
    {
        int occupancy = 0;
        err = cudaOccupancyMaxActiveBlocksPerMultiprocessor
        (
            &occupancy,
            recoverGasPrimitivesKernel,
            block,
            0
        );
        if (err != cudaSuccess || occupancy <= 0)
        {
            continue;
        }
        float median = 0.0f;
        if
        (
            measureToolB1Candidate
            (
                [=]() { return launchToolB1CellBundle(s, block); },
                start,
                stop,
                median,
                measuredTotal
            ) != 0
        )
        {
            cudaEventDestroy(stop);
            cudaEventDestroy(start);
            return 1;
        }
        if (bestCellBlock == 0 || median < bestCellMs)
        {
            bestCellBlock = block;
            bestCellMs = median;
        }

        if (s->hostTurbulenceModel == 0)
        {
            err = cudaOccupancyMaxActiveBlocksPerMultiprocessor
            (
                &occupancy,
                computeGasInternalFaceFluxKernel<false>,
                block,
                0
            );
        }
        else
        {
            err = cudaOccupancyMaxActiveBlocksPerMultiprocessor
            (
                &occupancy,
                computeGasInternalFaceFluxKernel<true>,
                block,
                0
            );
        }
        if (err != cudaSuccess || occupancy <= 0)
        {
            continue;
        }
        if
        (
            measureToolB1Candidate
            (
                [=]()
                {
                    return launchToolB1FaceBundle
                    (
                        s,
                        block,
                        dt,
                        simulationTime
                    );
                },
                start,
                stop,
                median,
                measuredTotal
            ) != 0
        )
        {
            cudaEventDestroy(stop);
            cudaEventDestroy(start);
            return 1;
        }
        if (bestFaceBlock == 0 || median < bestFaceMs)
        {
            bestFaceBlock = block;
            bestFaceMs = median;
        }
    }
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    if (bestCellBlock == 0 || bestFaceBlock == 0)
    {
        setLastErrorText("ToolB1 found no valid measured launch size");
        return 1;
    }
    s->fixedCellBlockThreads = bestCellBlock;
    s->fixedFaceBlockThreads = bestFaceBlock;
    s->fixedWorkBlockTuned = 1;
    std::fprintf
    (
        stderr,
        "ToolB1: B1cell=%d B1face=%d cellMedianMs=%.6f "
        "faceMedianMs=%.6f measuredKernelMs=%.6f warmup=%d repeats=%d\n",
        bestCellBlock,
        bestFaceBlock,
        bestCellMs,
        bestFaceMs,
        measuredTotal,
        toolB1WarmupRuns,
        toolB1MeasuredRuns
    );
    return 0;
}
