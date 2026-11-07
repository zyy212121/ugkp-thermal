#pragma once
#include "SteadyConfigurationSync.cuh"
#include "../../../common/GpuBatchedMaximum.cuh"

extern "C" int ugkwpGpuResidentStrictConfigureSteadySampling(void* handle,double boundaryTime)
{
    DeviceState* s=asState(handle);
    s->gasSteadyMode=false; // keep the complete original physical-dt scheme
    s->gasSteadyBoundaryFrozen=true;
    s->gasSteadyBoundaryTime=boundaryTime;
    // Sampling retains physical time steps; only boundary forcing is frozen.
    invalidateSharedGasConfiguration(s);
    const size_t n=size_t(GpuSteadySamplingReport::fields)*s->nCells;
    int rc=0;
    rc|=allocate(s->steadySampleSum,n,"allocate steady sample sum");
    rc|=allocate(s->steadySampleSquare,n,"allocate steady sample square");
    rc|=allocate(s->steadySampleMean,n,"allocate steady sample mean");
    rc|=allocate(s->steadySampleBegin,n,"allocate steady sample begin");
    rc|=allocate(s->steadySampleScratch,5*n,"allocate steady sample reduction");
    rc|=allocate(s->steadySampleReductionWorkspace,
        size_t(ugkwp::componentMaximumParts(s->nCells)+1)*5*GpuSteadySamplingReport::fields
            +GpuSteadySamplingReport::budgetFields,"allocate batched sample statistics");
    rc|=allocate(s->steadyBoundaryBudget,GpuSteadySamplingReport::budgetFields,"allocate steady boundary audit");
    if(rc)return rc;
    if(configureSteadyAuditStorage(s))return 1;
    auto err=cudaMemset(s->steadySampleMean,0,n*sizeof(double));
    if(err==cudaSuccess)err=cudaMemset(s->steadyBoundaryBudget,0,GpuSteadySamplingReport::budgetFields*sizeof(double));
    if(err!=cudaSuccess){setLastError("initialize steady sample storage",err);return 1;}
    if(syncSteadyMember(s,&DeviceState::gasSteadyMode,"sync steady gasSteadyMode"))return 1;
    if(syncSteadyMember(s,&DeviceState::gasSteadyBoundaryFrozen,"sync steady gasSteadyBoundaryFrozen"))return 1;
    if(syncSteadyMember(s,&DeviceState::gasSteadyBoundaryTime,"sync steady gasSteadyBoundaryTime"))return 1;
    if(syncSteadyMember(s,&DeviceState::gasGraphMode,"sync steady gasGraphMode"))return 1;
    if(syncSteadyMember(s,&DeviceState::steadySampleSum,"sync steady steadySampleSum"))return 1;
    if(syncSteadyMember(s,&DeviceState::steadySampleSquare,"sync steady steadySampleSquare"))return 1;
    if(syncSteadyMember(s,&DeviceState::steadySampleMean,"sync steady steadySampleMean"))return 1;
    if(syncSteadyMember(s,&DeviceState::steadySampleBegin,"sync steady steadySampleBegin"))return 1;
    if(syncSteadyMember(s,&DeviceState::steadySampleScratch,"sync steady steadySampleScratch"))return 1;
    if(syncSteadyMember(s,&DeviceState::steadySampleReductionWorkspace,"sync steady sample statistics"))return 1;
    if(syncSteadyMember(s,&DeviceState::steadyBoundaryBudget,"sync steady steadyBoundaryBudget"))return 1;
    return 0;
}

extern "C" int ugkwpGpuResidentStrictBeginSamplingWindow(void* handle)
{
    DeviceState* s=asState(handle);
    const int block=s->fixedCellBlockThreads,grid=(s->nCells+block-1)/block;
    beginSteadySampleWindowKernel<<<grid,block>>>(s->deviceState);
    auto err=cudaGetLastError();
    if(err==cudaSuccess)err=cudaMemset(s->steadyBoundaryBudget,0,GpuSteadySamplingReport::budgetFields*sizeof(double));
    if(err!=cudaSuccess){setLastError("begin steady sample window",err);return 1;}
    return 0;
}

extern "C" int ugkwpGpuResidentStrictAccumulateSamplingWindow(void* handle,double dt)
{
    DeviceState* s=asState(handle);
    const int block=s->fixedCellBlockThreads,grid=(s->nCells+block-1)/block;
    // Mixture transactions complete in advanceOneStep for both applications.
    // Retain the existing per-sample completion for the legacy particle path.
    if(s->gasSpecies.mode==ugkwp::GasMode::SingleLegacy
        && SharedGasTrialPolicy::complete(s))return 1;
    accumulateSteadySampleWindowKernel<<<grid,block>>>(s->deviceState,dt);
    const auto err=cudaGetLastError();
    if(err!=cudaSuccess){setLastError("accumulate steady sample window",err);return 1;}
    return 0;
}

extern "C" int ugkwpGpuResidentStrictFinishSamplingWindow
(void* handle,double duration,GpuSteadySamplingReport* report)
{
    DeviceState* s=asState(handle);
    const int block=s->fixedCellBlockThreads,grid=(s->nCells+block-1)/block;
    finishSteadySampleWindowKernel<<<grid,block>>>(s->deviceState,duration);
    const auto err=cudaGetLastError();
    if(err!=cudaSuccess){setLastError("finish steady sample window",err);return 1;}
    constexpr int fields=GpuSteadySamplingReport::fields,components=5*fields;
    auto status=ugkwp::batchedComponentStatistics(s->steadySampleScratch,s->nCells,
        components,2*fields,s->steadySampleReductionWorkspace,nullptr);
    double* result=s->steadySampleReductionWorkspace
        +size_t(ugkwp::componentMaximumParts(s->nCells))*components;
    if(status==cudaSuccess)status=cudaMemcpyAsync(result+components,s->steadyBoundaryBudget,
        GpuSteadySamplingReport::budgetFields*sizeof(double),cudaMemcpyDeviceToDevice);
    if(status!=cudaSuccess){setLastError("reduce steady sample statistics",status);return 1;}
    double values[components+GpuSteadySamplingReport::budgetFields];
    if(copyToHost(values,result,components+GpuSteadySamplingReport::budgetFields,
        "download steady sample statistics and boundary audit"))return 1;
    for(int k=0;k<fields;++k){
        report->meanChange[k]=values[k];report->fluctuation[k]=values[fields+k];
        report->inventoryBegin[k]=values[2*fields+k];report->inventoryEnd[k]=values[3*fields+k];
        report->characteristicInventory[k]=values[4*fields+k];
        report->inventoryChange[k]=report->inventoryEnd[k]-report->inventoryBegin[k];
    }
    const double* budget=values+components;
    for(int k=0;k<GpuSteadySamplingReport::gasFluxFields;++k)report->gasBoundaryExchange[k]=budget[k];
    for(int k=0;k<6;++k){report->particleInjection[k]=budget[GpuSteadySamplingReport::gasFluxFields+k];report->particleTrackingExchange[k]=budget[GpuSteadySamplingReport::gasFluxFields+6+k];}
    return SharedGasTrialPolicy::complete(s);
}

extern "C" int ugkwpGpuResidentStrictDownloadSamplingMean(void* handle,double* values)
{
    DeviceState* s=asState(handle);
    return copyToHost(values,s->steadySampleMean,size_t(GpuSteadySamplingReport::fields)*s->nCells,"download steady sample mean");
}
