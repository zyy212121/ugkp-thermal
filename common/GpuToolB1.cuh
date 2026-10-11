#pragma once
// Shared measurement/selection and checked event lifecycle.
// A successful zero-occupancy result is an unusable candidate; an API failure
// aborts tuning. No application identity or case-specific choice enters here.

#include "GpuToolB1Launch.cuh"

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
    const GasHostPolicy::Time dt,
    const GasHostPolicy::Time simulationTime
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
            recoverGasPrimitivesKernel<DeviceState>,
            block,
            0
        );
        if (err != cudaSuccess)
        {
            cudaEventDestroy(stop);
            cudaEventDestroy(start);
            setLastError("ToolB1 cell occupancy query", err);
            return 1;
        }
        if (occupancy <= 0)
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
                computeGasInternalFaceFluxKernel<false, DeviceState>,
                block,
                0
            );
        }
        else
        {
            err = cudaOccupancyMaxActiveBlocksPerMultiprocessor
            (
                &occupancy,
                computeGasInternalFaceFluxKernel<true, DeviceState>,
                block,
                0
            );
        }
        if (err != cudaSuccess)
        {
            cudaEventDestroy(stop);
            cudaEventDestroy(start);
            setLastError("ToolB1 face occupancy query", err);
            return 1;
        }
        if (occupancy <= 0)
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
