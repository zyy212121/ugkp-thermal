#include "../GpuGasCourantReduction.cuh"
#include "../GpuGasCourant.cuh"

extern "C" int ugkwpGpuResidentStrictComputeGasCourant
(
    void* handle, double dt, double scheduleTime, double* maxCo
)
{
    DeviceState* state=asState(handle);
    if(validateState(state,"compute gas Courant"))return 1;
    if(!maxCo){setLastErrorText("null gas Courant output");return 1;}
    const bool mixture=ugkwp::mixtureGasActive(*state);
    if(!mixture
#ifdef UGKWP_STEADY_APPLICATION
        && !state->gasSteadyBoundaryFrozen
#endif
        && publishStartupWorkPlan(state,dt,scheduleTime))return 1;

    const auto maximum=[](const GpuReal* acoustic,int count,cudaStream_t stream,double& value){
        try{value=ugkwp::reduceGasAcousticCourant(acoustic,count,stream);}
        catch(const std::exception& error){setLastErrorText(error.what());return 1;}
        const auto error=cudaGetLastError();
        if(error!=cudaSuccess){setLastError("gas Courant maximum reduction",error);return 1;}
        return 0;
    };
    GasCourantMaxima result;
    const auto stream=state->gasCaptureStream;
    int status=computeGasCurrentCourant(state,dt,scheduleTime,state->fixedCellBlockThreads,
        state->fixedFaceBlockThreads,stream,result,maximum);
    if(status&&!mixture)return 1;
    if(mixture&&!status)status=ResidentCourantTrialPolicy::validate(state);
    const auto error=state->diagnosticOwner?state->diagnosticOwner->observeCompletedWork(stream)
        :cudaStreamSynchronize(stream);
    if(error!=cudaSuccess){setLastError("diagnostics at Courant readback",error);return 1;}
    if(state->diagnosticOwner&&state->diagnosticOwner->dispatchPending())return 1;
    if(status||(mixture&&state->gasModelPoisoned))return 1;
    *maxCo=result.maximum;
    return 0;
}
