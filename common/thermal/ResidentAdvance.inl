extern "C" int ugkwpGpuResidentStrictAdvance
(
    void* handle,
    double dt,
    double simulationTime
)
{



    DeviceState* s = asState(handle);
    if (validateState(s, "advance") != 0)
    {
        return 1;
    }


    if(ugkwp::mixtureGasActive(*s))return advanceChtSharedGas(s,dt,simulationTime);
    if (publishStartupWorkPlan(s, dt, simulationTime-dt) != 0)
    {
        return 1;
    }


#ifndef UGKP_DEVELOPMENT_PROBES
    if (s->gasGraphMode < 0)
    {
        const char* value = std::getenv("UGKP_GAS_GRAPH");
        s->gasGraphMode = !value || std::strcmp(value, "1") == 0;
    }
    if (s->gasGraphMode && !s->particlesMayBePresent && !s->gravityEnabled)
    {
        if(advancePureGasGraph(s,dt,simulationTime-dt))return 1;
        return ChtSpeciesTrialPolicy::commitInterval(s);
    }
#endif

    const int block = s->reductionBlockThreads;
    const int warpCount = (block + 31)/32;
    const int grid = (s->nCells + block - 1)/block;
    cudaError_t err = cudaSuccess;

#ifdef UGKP_DEVELOPMENT_PROBES
    DevelopmentAdvanceProbe developmentAdvanceProbe(s, dt, simulationTime);
    if (developmentAdvanceProbe.failed())
    {
        return 1;
    }
#define UGKP_DEV_PROBE_ENTER(STAGE) developmentAdvanceProbe.enter(STAGE)
#define UGKP_DEV_PROBE_LEAVE(STAGE) \
    do \
    { \
        if (developmentAdvanceProbe.leave(STAGE) != 0) \
        { \
            return 1; \
        } \
    } while (false)
#else
#define UGKP_DEV_PROBE_ENTER(STAGE) ((void)0)
#define UGKP_DEV_PROBE_LEAVE(STAGE) ((void)0)
#endif

    UGKP_DEV_PROBE_ENTER(ProbeGasFlux);
    if
    (
        s->particlesMayBePresent
     && s->particleStuckModelConfigured != 0
     && s->particleWorkGrid > 0
     && prepareParticleWallContactAreaScale(s, block, s->gasCaptureStream) != 0
    )
    {
        return 1;
    }
    if (advanceGasFluxScheduled(s, dt, simulationTime-dt) != 0)
    {
        return 1;
    }
    if (s->gravityEnabled != 0)
    {
        const int gasBlock = s->fixedCellBlockThreads;
        const int gasGrid = (s->nCells + gasBlock - 1)/gasBlock;
        applyGasGravityKernel<<<gasGrid, gasBlock,0,s->gasCaptureStream>>>(s->deviceState, dt);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("applyGasGravityKernel launch", err);
            return 1;
        }
        recoverGasPrimitivesKernel<<<gasGrid, gasBlock,0,s->gasCaptureStream>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError
            (
                "recoverGasPrimitivesKernel after gravity launch",
                err
            );
            return 1;
        }
    }

    UGKP_DEV_PROBE_LEAVE(ProbeGasFlux);

    const bool skipParticlePath = !s->particlesMayBePresent;
    if(advanceResidentThermalParticles(s,dt,simulationTime
#ifdef UGKP_DEVELOPMENT_PROBES
        ,&developmentAdvanceProbe
#endif
    )!=0)return 1;

    UGKP_DEV_PROBE_ENTER(ProbeBoundary);
    if (finaliseGasBoundaryStage(s, dt, simulationTime) != 0)
    {
        return 1;
    }

    if(ChtSpeciesTrialPolicy::commitInterval(s))return 1;
    UGKP_DEV_PROBE_LEAVE(ProbeBoundary);

#ifdef UGKP_DEVELOPMENT_PROBES
    if (developmentAdvanceProbe.finish(!skipParticlePath) != 0)
    {
        return 1;
    }
#endif

#undef UGKP_DEV_PROBE_ENTER
#undef UGKP_DEV_PROBE_LEAVE

    return 0;
}
