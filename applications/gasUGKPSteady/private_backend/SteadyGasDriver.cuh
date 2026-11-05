#pragma once
#include "SteadyConfigurationSync.cuh"
#include "SteadyGasKernels.cuh"
#include "SteadyValidationExport.cuh"

extern "C" int ugkwpGpuResidentStrictConfigureSteady(void* handle,double boundaryTime)
{
    DeviceState* s=asState(handle);
    s->gasSteadyMode=true;
    s->gasSteadyBoundaryFrozen=true;
    s->gasSteadyBoundaryTime=boundaryTime;
    s->gasPseudoReferenceDt=1;
    // Steady refresh has its own step-dependent launch arguments. Do not reuse
    // a transient graph or autotune by performing an unreported pseudo step.
    s->gasGraphMode=0;
    if(!s->gasPseudoDt&&allocate(s->gasPseudoDt,s->nCells,"allocate local pseudo time"))return 1;
    if(!s->gasSteadyResidual&&allocate(s->gasSteadyResidual,
        size_t(s->nCells)*(7+ugkwp::compiledGasSpecies),"allocate steady residual"))return 1;
    const auto error=cudaMemset(s->gasPseudoDt,0,s->nCells*sizeof(double));
    if(error!=cudaSuccess){setLastError("initialize local pseudo time",error);return 1;}
    if(syncSteadyMember(s,&DeviceState::gasSteadyMode,"sync steady gasSteadyMode"))return 1;
    if(syncSteadyMember(s,&DeviceState::gasSteadyBoundaryFrozen,"sync steady gasSteadyBoundaryFrozen"))return 1;
    if(syncSteadyMember(s,&DeviceState::gasSteadyBoundaryTime,"sync steady gasSteadyBoundaryTime"))return 1;
    if(syncSteadyMember(s,&DeviceState::gasPseudoReferenceDt,"sync steady gasPseudoReferenceDt"))return 1;
    if(syncSteadyMember(s,&DeviceState::gasGraphMode,"sync steady gasGraphMode"))return 1;
    if(syncSteadyMember(s,&DeviceState::gasPseudoDt,"sync steady gasPseudoDt"))return 1;
    if(syncSteadyMember(s,&DeviceState::gasSteadyResidual,"sync steady gasSteadyResidual"))return 1;
    return 0;
}

extern "C" int ugkwpGpuResidentStrictPrepareSteadyStep(void* handle,double courant,double maximumDt)
{
    DeviceState* s=asState(handle);
    double maximumCo=0;
    if(ugkwpGpuResidentStrictComputeGasCourant(handle,1,s->gasSteadyBoundaryTime,&maximumCo))return 1;
    const int block=s->fixedCellBlockThreads,grid=(s->nCells+block-1)/block;
    prepareSteadyGasTimeStepKernel<<<grid,block,0,s->gasCaptureStream>>>(s->deviceState,courant,maximumDt);
    const auto error=cudaGetLastError();
    if(error!=cudaSuccess){setLastError("prepare steady local time",error);return 1;}
    return 0;
}

extern "C" int ugkwpGpuResidentStrictSteadyResidual(void* handle,double* residual)
{
    DeviceState* s=asState(handle);
    ugkwp::invalidateGasBoundaryLayerPreparation(*s);
    const int faceBlock=s->fixedFaceBlockThreads;
    const int boundaryGrid=(s->nFaces-s->nInternalFaces+faceBlock-1)/faceBlock;
    if(boundaryGrid>0)
        updateLegacyGasBoundaryMirrorKernel<<<boundaryGrid,faceBlock,0,s->gasCaptureStream>>>
            (s->deviceState,s->gasSteadyBoundaryTime);
    // dt=0 refreshes raw face fluxes without a time-weighted positivity limiter.
    // It applies the same algebraic wall constraints/primitive projections.
    if(advanceGasEulerSubstage(s,0,0,s->gasSteadyBoundaryTime))return 1;
    const int block=s->fixedCellBlockThreads,grid=(s->nCells+block-1)/block;
    steadyGasSpatialResidualKernel<<<grid,block,0,s->gasCaptureStream>>>(s->deviceState);
    const auto error=cudaGetLastError();
    if(error!=cudaSuccess){setLastError("steady spatial residual",error);return 1;}
    try
    {
        for(int k=0;k<7+ugkwp::compiledGasSpecies;++k)
            residual[k]=ugkwp::reduceGasAcousticCourant(
                s->gasSteadyResidual+k*s->nCells,s->nCells,s->gasCaptureStream);
    }
    catch(const std::exception& e){setLastErrorText(e.what());return 1;}
    exportSteadyValidationSnapshot(s);
    return SharedGasTrialPolicy::complete(s);
}
