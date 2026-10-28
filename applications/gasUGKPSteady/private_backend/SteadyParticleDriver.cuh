#pragma once
#include "SteadyConfigurationSync.cuh"

extern "C" int ugkwpGpuResidentStrictConfigureSteadySampling(void* handle,double boundaryTime)
{
    DeviceState* s=asState(handle);
    s->gasSteadyMode=false; // keep the complete original physical-dt scheme
    s->gasSteadyBoundaryFrozen=true;
    s->gasSteadyBoundaryTime=boundaryTime;
    s->gasGraphMode=0;
    const size_t n=size_t(GpuSteadySamplingReport::fields)*s->nCells;
    int rc=0;
    rc|=allocate(s->steadySampleSum,n,"allocate steady sample sum");
    rc|=allocate(s->steadySampleSquare,n,"allocate steady sample square");
    rc|=allocate(s->steadySampleMean,n,"allocate steady sample mean");
    rc|=allocate(s->steadySampleBegin,n,"allocate steady sample begin");
    rc|=allocate(s->steadySampleScratch,5*n,"allocate steady sample reduction");
    rc|=allocate(s->steadyBoundaryBudget,GpuSteadySamplingReport::budgetFields,"allocate steady boundary audit");
    if(rc)return rc;
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
    // Consume the physical step's diagnostics before accepting its sample or
    // letting the frontend advance sampling time and publish restart fields.
    if(SharedGasTrialPolicy::complete(s))return 1;
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
    const int n=GpuSteadySamplingReport::fields*s->nCells;
    try
    {
        for(int k=0;k<GpuSteadySamplingReport::fields;++k)
        {
            report->meanChange[k]=ugkwp::reduceGasAcousticCourant(s->steadySampleScratch+k*s->nCells,s->nCells,nullptr);
            report->fluctuation[k]=ugkwp::reduceGasAcousticCourant(s->steadySampleScratch+n+k*s->nCells,s->nCells,nullptr);
            auto begin=thrust::device_pointer_cast(s->steadySampleScratch+2*n+k*s->nCells);
            auto end=thrust::device_pointer_cast(s->steadySampleScratch+3*n+k*s->nCells);
            auto characteristic=thrust::device_pointer_cast(s->steadySampleScratch+4*n+k*s->nCells);
            report->characteristicInventory[k]=thrust::reduce(characteristic,characteristic+s->nCells,0.0,thrust::plus<double>());
            report->inventoryBegin[k]=thrust::reduce(begin,begin+s->nCells,0.0,thrust::plus<double>());
            report->inventoryEnd[k]=thrust::reduce(end,end+s->nCells,0.0,thrust::plus<double>());
            report->inventoryChange[k]=report->inventoryEnd[k]-report->inventoryBegin[k];
        }
    }
    catch(const std::exception& e){setLastErrorText(e.what());return 1;}
    double budget[GpuSteadySamplingReport::budgetFields];
    if(copyToHost(budget,s->steadyBoundaryBudget,GpuSteadySamplingReport::budgetFields,"download steady boundary audit"))return 1;
    for(int k=0;k<GpuSteadySamplingReport::gasFluxFields;++k)report->gasBoundaryExchange[k]=budget[k];
    for(int k=0;k<6;++k){report->particleInjection[k]=budget[GpuSteadySamplingReport::gasFluxFields+k];report->particleTrackingExchange[k]=budget[GpuSteadySamplingReport::gasFluxFields+6+k];}
    return SharedGasTrialPolicy::complete(s);
}

extern "C" int ugkwpGpuResidentStrictDownloadSamplingMean(void* handle,double* values)
{
    DeviceState* s=asState(handle);
    return copyToHost(values,s->steadySampleMean,size_t(GpuSteadySamplingReport::fields)*s->nCells,"download steady sample mean");
}
