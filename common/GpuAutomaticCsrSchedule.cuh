#pragma once
#include "GpuParticleExecutionContext.H"
#include "operators/maximumDirectoryOccupancyKernel.cuh"
#include "operators/publishHeavyReductionDecisionKernel.cuh"

int prepareCsrSegmentedReductionTasks
(DeviceState* s, const int block,
 const GPU_DIRECTORY_PARAMETER_TYPE GPU_DIRECTORY_SELECTOR);
int prepareCsrSegmentedReductionTasksOnInspection
(DeviceState* s,const int block,const GPU_DIRECTORY_PARAMETER_TYPE GPU_DIRECTORY_SELECTOR);

#ifdef UGKP_RESIDENT_CSR_CHOICE
// The inspection cadence belongs to the device schedule, including graph replay.
__global__ void beginAutomaticCsrInspectionKernel(DeviceState* s) {
    if(threadIdx.x||blockIdx.x)return;
    auto& runtime=*s->csrAutomaticRuntime;
    ++runtime.advanceCount;
    runtime.inspect=runtime.advanceCount==1u
        ||runtime.advanceCount%static_cast<unsigned long long>(s->csrHeavyAutoInterval)==0u;
    if(runtime.inspect)*s->csrMaximumOccupancy=0;
}
__global__ void decideAutomaticCsrScheduleKernel(DeviceState* s) {
    if(threadIdx.x||blockIdx.x)return;
    if(!s->csrAutomaticRuntime->inspect)return;
    const int active=*s->csrMaximumOccupancy>s->GPU_AUTO_THRESHOLD_FIELD?1:0;
    s->csrAutomaticRuntime->active=active;
    s->csrHeavyReductionActive=active;s->csrHeavyReductionEnabled=active;
}
#endif

// A successful auto call publishes a decision AND guarantees executable tasks.
// Producers invalidate readiness in the common task pipeline, including while
// L1 is active. Normal non-inspection steps reuse their freshly prepared tasks.
int runAutomaticCsrSchedule
(DeviceState* s, const int block,
 const GPU_DIRECTORY_PARAMETER_TYPE GPU_DIRECTORY_SELECTOR)
{
    if (s->csrHeavyReductionMode != 2 || s->particleCapacity <= 0) return 0;
    if (s->csrHeavyAutoInterval == 0 || block < 1)
    {
        setLastError("automatic CSR schedule invalid parameters", cudaErrorInvalidValue);
        return 1;
    }
#ifdef UGKP_RESIDENT_CSR_CHOICE
    const auto stream=ugkp::gpu::particleRecordingStream(s);
    beginAutomaticCsrInspectionKernel<<<1,1,0,stream>>>(s->deviceState);
    auto err=cudaGetLastError();
    if(err!=cudaSuccess){setLastError("begin automatic CSR inspection",err);return 1;}
    if(GPU_AUTO_INSPECTION_UPDATE_POLICY(s,GPU_DIRECTORY_SELECTOR)!=0)return 1;
    const int grid=(s->nCells+block-1)/block;
    maximumDirectoryOccupancyKernel<<<grid,block,0,stream>>>
        (s->deviceState,GPU_DIRECTORY_ARGUMENT,s->csrMaximumOccupancy,true);
    err=cudaGetLastError();
    if(err==cudaSuccess){decideAutomaticCsrScheduleKernel<<<1,1,0,stream>>>(s->deviceState);err=cudaGetLastError();}
    if(err!=cudaSuccess){setLastError("automatic CSR resident decision",err);return 1;}
    // A fresh directory already has tasks for non-inspection steps. Rebuild
    // only if the device inspection changed the threshold/selected path.
    if(s->csrTasksReady&&s->csrPreparedDirectoryKind==GPU_DIRECTORY_ARGUMENT)
        return prepareCsrSegmentedReductionTasksOnInspection(s,block,GPU_DIRECTORY_SELECTOR);
    s->csrTasksReady=0;
#else
    ++s->schedulingAdvanceCount;
    const bool inspect = s->schedulingAdvanceCount == 1u
        || s->schedulingAdvanceCount
          % static_cast<unsigned long long>(s->csrHeavyAutoInterval) == 0u;
    if (inspect)
    {
        if (GPU_AUTO_UPDATE_POLICY(s, GPU_DIRECTORY_SELECTOR) != 0) return 1;
        cudaError_t err = cudaMemsetAsync(s->csrMaximumOccupancy, 0, sizeof(int));
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
            err = cudaMemsetAsync(s->csrHeavyTaskCount, 0, sizeof(int));
            if (err == cudaSuccess)
                err = cudaMemsetAsync(s->csrHeavyCellCount, 0, sizeof(int));
            if (err != cudaSuccess)
            {
                setLastError("clear inactive automatic CSR schedule", err);
                return 1;
            }
        }
    }
#endif
    if (s->csrHeavyReductionEnabled != 0
        && (s->csrTasksReady == 0
            || s->csrPreparedDirectoryKind != GPU_DIRECTORY_ARGUMENT))
        return prepareCsrSegmentedReductionTasks(s, block, GPU_DIRECTORY_SELECTOR);
    return 0;
}
