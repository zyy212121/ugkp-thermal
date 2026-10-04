#pragma once
#include "operators/maximumDirectoryOccupancyKernel.cuh"
#include "operators/publishHeavyReductionDecisionKernel.cuh"

int prepareCsrSegmentedReductionTasks
(DeviceState* s, const int block,
 const GPU_DIRECTORY_PARAMETER_TYPE GPU_DIRECTORY_SELECTOR);

// A successful auto call publishes a decision AND guarantees executable tasks.
// Producers invalidate readiness in the common task pipeline, including while
// L1 is active. Normal non-inspection steps reuse their freshly prepared tasks.
int runAutomaticCsrSchedule
(DeviceState* s, const int block,
 const GPU_DIRECTORY_PARAMETER_TYPE GPU_DIRECTORY_SELECTOR)
{
    if (s->csrHeavyReductionMode != 2 || s->particleCapacity <= 0) return 0;
    if (s->csrHeavyAutoInterval < 1 || block < 1 || s->nCells < 1)
    {
        setLastError("automatic CSR schedule invalid parameters", cudaErrorInvalidValue);
        return 1;
    }
    ++s->schedulingAdvanceCount;
    const bool inspect = s->schedulingAdvanceCount == 1u
        || s->schedulingAdvanceCount
          % static_cast<unsigned long long>(s->csrHeavyAutoInterval) == 0u;
    if (inspect)
    {
        if (GPU_AUTO_UPDATE_POLICY(s, GPU_DIRECTORY_SELECTOR) != 0) return 1;
        cudaError_t err = cudaMemset(s->csrMaximumOccupancy, 0, sizeof(int));
        if (err == cudaSuccess)
        {
            const int grid = (s->nCells + block - 1)/block;
            maximumDirectoryOccupancyKernel<<<grid, block>>>
            (s->deviceState, GPU_DIRECTORY_ARGUMENT, s->csrMaximumOccupancy);
            err = cudaGetLastError();
        }
        int maximumOccupancy = 0;
        int threshold = 0;
        if (err == cudaSuccess)
            err = cudaMemcpy(&maximumOccupancy, s->csrMaximumOccupancy,
                sizeof(int), cudaMemcpyDeviceToHost);
        if (err == cudaSuccess)
            err = cudaMemcpy(&threshold,
                reinterpret_cast<const unsigned char*>(s->deviceState)
                  + offsetof(DeviceState, GPU_AUTO_THRESHOLD_FIELD),
                sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            setLastError("automatic CSR occupancy decision", err);
            return 1;
        }
        const int active = maximumOccupancy > threshold ? 1 : 0;
        s->GPU_AUTO_THRESHOLD_FIELD = threshold;
        s->csrHeavyTileParticles = threshold;
        s->csrHeavyReductionActive = active;
        s->csrHeavyReductionEnabled = active;
        // A fresh threshold invalidates any earlier task partition.
        s->csrTasksReady = 0;
        publishHeavyReductionDecisionKernel<<<1, 1>>>(s->deviceState, active);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("automatic CSR decision publication", err);
            return 1;
        }
        if (active == 0)
        {
            err = cudaMemset(s->csrHeavyTaskCount, 0, sizeof(int));
            if (err == cudaSuccess)
                err = cudaMemset(s->csrHeavyCellCount, 0, sizeof(int));
            if (err != cudaSuccess)
            {
                setLastError("clear inactive automatic CSR schedule", err);
                return 1;
            }
        }
    }
    if (s->csrHeavyReductionEnabled != 0
        && (s->csrTasksReady == 0
            || s->csrPreparedDirectoryKind != GPU_DIRECTORY_ARGUMENT))
        return prepareCsrSegmentedReductionTasks(s, block, GPU_DIRECTORY_SELECTOR);
    return 0;
}
