#pragma once
#include <type_traits>
#include "gasTransport/GasStateView.H"
#include "operators/advanceGasChemistryKernel.cuh"

// Shared Euler/RK, boundary-finalisation and graph-capture host protocol.
// Requires a gas host state, GasHostPolicy, all gas kernels and error helpers.
// The host state owns launch/graph resources and a deviceState pointer. That
// pointer may name a gas-only view; no particle storage is required here.
// GasHostPolicy owns only time/weight types and the optional wall-energy ledger;
// launches, streams, error checks and their ordering are owned here.
// RK fractions are converted before division to preserve FP32 weight semantics.
template<class GasHostState>
int advanceGasEulerSubstage(GasHostState* s, const GasHostPolicy::Time dt, const GasHostPolicy::Time ledgerDt, const GasHostPolicy::Time stageTime)
{
    const int cellBlock = s->fixedCellBlockThreads;
    const int faceBlock = s->fixedFaceBlockThreads;
    const int cellGrid = (s->nCells + cellBlock - 1)/cellBlock;
    const int faceGrid = (s->nFaces + faceBlock - 1)/faceBlock;
    cudaError_t err = cudaSuccess;

    recoverGasPrimitivesKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("recoverGasPrimitivesKernel gas stage", err);
        return 1;
    }
    if (s->hostTurbulenceModel == 3)
    {
        recoverSstPrimitivesKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("recoverSstPrimitivesKernel gas stage", err);
            return 1;
        }
    }
    if (faceGrid > 0)
    {
        updateRiemannBoundaryMirrorKernel<<<faceGrid, faceBlock, 0, s->gasCaptureStream>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("updateRiemannBoundaryMirrorKernel gas stage", err);
            return 1;
        }
    }
    if (s->hostTurbulenceModel == 3)
    {
        applySstWallFunctionStateKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("applySstWallFunctionStateKernel gas stage", err);
            return 1;
        }
    }
    if (s->hostGasFluxScheme == 7)
    {
        computeGasHllcAdcSensorKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("computeGasHllcAdcSensorKernel launch", err);
            return 1;
        }
    }
    computeGasPrimitiveGradientsKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("computeGasPrimitiveGradientsKernel launch", err);
        return 1;
    }
    computeGasGradientLimiterKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("computeGasGradientLimiterKernel launch", err);
        return 1;
    }
    computeGasEddyViscosityKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("computeGasEddyViscosityKernel launch", err);
        return 1;
    }
    if (faceGrid <= 0)
    {
        return 0;
    }
    if (s->hostTurbulenceModel == 0)
    {
        computeGasInternalFaceFluxKernel<false><<<faceGrid, faceBlock, 0, s->gasCaptureStream>>>
        (
            s->deviceState,
            dt
        );
    }
    else
    {
        computeGasInternalFaceFluxKernel<true><<<faceGrid, faceBlock, 0, s->gasCaptureStream>>>
        (
            s->deviceState,
            dt
        );
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("computeGasInternalFaceFluxKernel launch", err);
        return 1;
    }
    if (s->hasPeriodicFaces != 0)
    {
        enforcePeriodicGasFluxAntisymmetryKernel<<<faceGrid, faceBlock, 0, s->gasCaptureStream>>>
        (
            s->deviceState
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("enforcePeriodicGasFluxAntisymmetryKernel launch", err);
            return 1;
        }
    }
    // Coupled applications may replace complete 5+Ns boundary fluxes here.
    // Positivity and accepted RK-weighted ledgers still belong to this pipeline.
    if (GasHostPolicy::applyFaceSources(s, dt, stageTime) != 0) return 1;
    computeGasFluxPositivityScaleKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState, dt);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("computeGasFluxPositivityScaleKernel launch", err);
        return 1;
    }
    applyGasFluxPositivityScaleKernel<<<faceGrid, faceBlock, 0, s->gasCaptureStream>>>
    (
        s->deviceState
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("applyGasFluxPositivityScaleKernel launch", err);
        return 1;
    }
    if (GasHostPolicy::accumulateWallEnergy(s, ledgerDt) != 0) return 1;
    if (s->hasPeriodicFaces != 0)
    {
        enforcePeriodicGasFluxAntisymmetryKernel<<<faceGrid, faceBlock, 0, s->gasCaptureStream>>>
        (
            s->deviceState
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError
            (
                "enforcePeriodicGasFluxAntisymmetryKernel post-scale launch",
                err
            );
            return 1;
        }
    }
    if (s->hostTurbulenceModel == 3)
    {
        computeSstGradientsKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("computeSstGradientsKernel launch", err);
            return 1;
        }
    }
    if (s->hostTurbulenceModel == 3)
    {
        computeSstFaceFluxKernel<<<faceGrid, faceBlock, 0, s->gasCaptureStream>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("computeSstFaceFluxKernel launch", err);
            return 1;
        }
        if (s->hasPeriodicFaces != 0)
        {
            enforcePeriodicSstFluxAntisymmetryKernel<<<faceGrid, faceBlock, 0, s->gasCaptureStream>>>
            (
                s->deviceState
            );
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                setLastError("enforcePeriodicSstFluxAntisymmetryKernel launch", err);
                return 1;
            }
        }
        applySstFluxAndSourceKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>
        (
            s->deviceState,
            dt
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("applySstFluxAndSourceKernel launch", err);
            return 1;
        }
    }
    applyGasFluxDivergenceByCellKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState, dt);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("applyGasFluxDivergenceByCellKernel launch", err);
        return 1;
    }
    recoverGasPrimitivesKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("recoverGasPrimitivesKernel post-gas launch", err);
        return 1;
    }
    if (s->hostTurbulenceModel == 3)
    {
        recoverSstPrimitivesKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("recoverSstPrimitivesKernel post-gas launch", err);
            return 1;
        }
    }
    return 0;
}

template<class GasHostState>
int blendGasRungeKuttaStage
(
    GasHostState* s,
    const GasHostPolicy::Weight initialWeight,
    const GasHostPolicy::Weight stageWeight
)
{
    const int block = s->fixedCellBlockThreads;
    const int cellGrid = (s->nCells + block - 1)/block;
    blendGasConservativeStateKernel<<<cellGrid, block, 0, s->gasCaptureStream>>>
    (
        s->deviceState,
        initialWeight,
        stageWeight
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("blendGasConservativeStateKernel launch", err);
        return 1;
    }
    recoverGasPrimitivesKernel<<<cellGrid, block, 0, s->gasCaptureStream>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("recoverGasPrimitivesKernel after RK blend", err);
        return 1;
    }
    if (s->hostTurbulenceModel == 3)
    {
        recoverSstPrimitivesKernel<<<cellGrid, block, 0, s->gasCaptureStream>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("recoverSstPrimitivesKernel after RK blend", err);
            return 1;
        }
    }
    return 0;
}

template<class GasHostState>
int advanceGasFluxStage(GasHostState* s, const GasHostPolicy::Time dt, const GasHostPolicy::Time simulationTime)
{

    const int preparationCellBlock = s->fixedCellBlockThreads;
    const int preparationFaceBlock = s->fixedFaceBlockThreads;
    const int preparationCellGrid =
        (s->nCells + preparationCellBlock - 1)/preparationCellBlock;
    const int preparationFaceGrid =
        (s->nFaces + preparationFaceBlock - 1)/preparationFaceBlock;
    using GasDevice=typename std::remove_pointer<decltype(s->deviceState)>::type;
    if constexpr(ugkwp::GasSstAuditCapability<GasDevice>::value)
    {
        prepareGasSstAuditKernel<<<preparationCellGrid,preparationCellBlock,0,s->gasCaptureStream>>>(s->deviceState);
        const auto error=cudaGetLastError();
        if(error!=cudaSuccess){setLastError("prepare SST stage audit",error);return 1;}
    }
    recoverGasPrimitivesKernel<<<preparationCellGrid, preparationCellBlock, 0, s->gasCaptureStream>>>
    (
        s->deviceState
    );
    cudaError_t preparationError = cudaGetLastError();
    if (preparationError != cudaSuccess)
    {
        setLastError
        (
            "recoverGasPrimitivesKernel before gas advance",
            preparationError
        );
        return 1;
    }
    if (preparationFaceGrid > 0)
    {
        updateLegacyGasBoundaryMirrorKernel
            <<<preparationFaceGrid, preparationFaceBlock, 0, s->gasCaptureStream>>>(s->deviceState, simulationTime);
        preparationError = cudaGetLastError();
        if (preparationError != cudaSuccess)
        {
            setLastError
            (
                "updateLegacyGasBoundaryMirrorKernel before gas advance",
                preparationError
            );
            return 1;
        }
    }

    if (s->hostGasTimeIntegrator == 1)
    {
        return advanceGasEulerSubstage(s, dt, dt, simulationTime);
    }

    const int block = s->fixedCellBlockThreads;
    const int cellGrid = (s->nCells + block - 1)/block;
    saveGasConservativeStateKernel<<<cellGrid, block, 0, s->gasCaptureStream>>>(s->deviceState);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("saveGasConservativeStateKernel launch", err);
        return 1;
    }

    const GasHostPolicy::Time firstLedgerDt = GasHostPolicy::firstLedgerDt(s, dt);
    if (advanceGasEulerSubstage(s, dt, firstLedgerDt, simulationTime) != 0)
    {
        return 1;
    }
    if (advanceGasEulerSubstage(s, dt, firstLedgerDt, simulationTime+dt) != 0)
    {
        return 1;
    }

    if (s->hostGasTimeIntegrator == 2)
    {

        return blendGasRungeKuttaStage(s, static_cast<GasHostPolicy::Weight>(0.5), static_cast<GasHostPolicy::Weight>(0.5));
    }
    if (s->hostGasTimeIntegrator != 3)
    {
        setLastErrorText("invalid resident gas time-integrator selector");
        return 1;
    }

    if (blendGasRungeKuttaStage(s, static_cast<GasHostPolicy::Weight>(0.75), static_cast<GasHostPolicy::Weight>(0.25)) != 0)
    {
        return 1;
    }
    if (advanceGasEulerSubstage(s, dt, GasHostPolicy::finalLedgerDt(dt), simulationTime+dt*GasHostPolicy::Time(0.5)) != 0)
    {
        return 1;
    }
    return blendGasRungeKuttaStage
    (
        s,
        static_cast<GasHostPolicy::Weight>(1.0)/static_cast<GasHostPolicy::Weight>(3.0),
        static_cast<GasHostPolicy::Weight>(2.0)/static_cast<GasHostPolicy::Weight>(3.0)
    );
}

template<class GasHostState>
int finaliseGasBoundaryStage(GasHostState* s, const GasHostPolicy::Time dt, const GasHostPolicy::Time simulationTime)
{
    const int block = s->fixedFaceBlockThreads;
    const int allFaceGrid = (s->nFaces + block - 1)/block;
    if (allFaceGrid <= 0)
    {
        return 0;
    }

    updateLegacyGasBoundaryMirrorKernel<<<allFaceGrid, block, 0, s->gasCaptureStream>>>
    (
        s->deviceState,
        simulationTime
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError
        (
            "updateLegacyGasBoundaryMirrorKernel final gas boundary",
            err
        );
        return 1;
    }

    updateRiemannBoundaryMirrorKernel<<<allFaceGrid, block, 0, s->gasCaptureStream>>>
    (
        s->deviceState
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError
        (
            "updateRiemannBoundaryMirrorKernel final gas boundary",
            err
        );
        return 1;
    }

    updateWaveTransmissivePressureBoundaryKernel<<<allFaceGrid, block, 0, s->gasCaptureStream>>>
    (
        s->deviceState,
        dt
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError
        (
            "updateWaveTransmissivePressureBoundaryKernel final gas boundary",
            err
        );
        return 1;
    }
    return 0;
}

template<class GasHostState>
int advancePureGasGraph(GasHostState* s, const GasHostPolicy::Time dt, const GasHostPolicy::Time simulationTime)
{
    if (s->gasGraphExec == nullptr || s->gasGraphDt != dt)
    {
        cudaError_t err = cudaSuccess;
        if (s->gasGraphExec)
        {
            err = cudaGraphExecDestroy(s->gasGraphExec);
            if (err != cudaSuccess) { setLastError("gas graph destroy executable", err); return 1; }
            s->gasGraphExec = nullptr;
        }
        if (s->gasGraph)
        {
            err = cudaGraphDestroy(s->gasGraph);
            if (err != cudaSuccess) { setLastError("gas graph destroy", err); return 1; }
            s->gasGraph = nullptr;
        }
        s->gasGraphTimeNodes.clear();
        s->gasGraphTimeParams.clear();
        err = cudaStreamCreate(&s->gasCaptureStream);
        if (err != cudaSuccess) { setLastError("gas graph stream create", err); return 1; }
        err = cudaStreamBeginCapture(s->gasCaptureStream, cudaStreamCaptureModeThreadLocal);
        if (err != cudaSuccess)
        {
            cudaStreamDestroy(s->gasCaptureStream);
            s->gasCaptureStream = nullptr;
            setLastError("gas graph capture begin", err);
            return 1;
        }
        int status = advanceGasFluxStage(s, dt, simulationTime);
        if (status == 0) status = finaliseGasBoundaryStage(s, dt, simulationTime);
        err = cudaStreamEndCapture(s->gasCaptureStream, &s->gasGraph);
        const cudaError_t streamError = cudaStreamDestroy(s->gasCaptureStream);
        s->gasCaptureStream = nullptr;
        if (status != 0) return status;
        if (err != cudaSuccess) { setLastError("gas graph capture end", err); return 1; }
        if (streamError != cudaSuccess) { setLastError("gas graph stream destroy", streamError); return 1; }
        size_t nodeCount = 0;
        err = cudaGraphGetNodes(s->gasGraph, nullptr, &nodeCount);
        if (err != cudaSuccess) { setLastError("gas graph node count", err); return 1; }
        std::vector<cudaGraphNode_t> nodes(nodeCount);
        err = cudaGraphGetNodes(s->gasGraph, nodes.data(), &nodeCount);
        if (err != cudaSuccess) { setLastError("gas graph nodes", err); return 1; }
        for (const auto node : nodes)
        {
            cudaGraphNodeType type;
            err = cudaGraphNodeGetType(node, &type);
            if (err != cudaSuccess) { setLastError("gas graph node type", err); return 1; }
            if (type != cudaGraphNodeTypeKernel) continue;
            cudaKernelNodeParams params{};
            err = cudaGraphKernelNodeGetParams(node, &params);
            if (err != cudaSuccess) { setLastError("gas graph kernel parameters", err); return 1; }
            using GasDeviceState = typename std::remove_pointer<decltype(s->deviceState)>::type;
            if (params.func == reinterpret_cast<void*>(updateLegacyGasBoundaryMirrorKernel<GasDeviceState>))
            {
                s->gasGraphTimeNodes.push_back(node);
                params.kernelParams = nullptr;
                params.extra = nullptr;
                s->gasGraphTimeParams.push_back(params);
            }
        }
        if (s->nFaces > 0 && s->gasGraphTimeNodes.size() != 2)
        {
            setLastErrorText("gas graph requires both scheduled boundary time nodes");
            return 1;
        }
        err = cudaGraphInstantiate(&s->gasGraphExec, s->gasGraph, nullptr, nullptr, 0);
        if (err != cudaSuccess) { setLastError("gas graph instantiate", err); return 1; }
        s->gasGraphDt = dt;
    }
    auto* device = s->deviceState;
    GasHostPolicy::Time currentTime = simulationTime;
    void* args[] = {&device, &currentTime};
    for (size_t i = 0; i < s->gasGraphTimeNodes.size(); ++i)
    {
        cudaKernelNodeParams params = s->gasGraphTimeParams[i];
        params.kernelParams = args;
        const cudaError_t err = cudaGraphExecKernelNodeSetParams
        (
            s->gasGraphExec, s->gasGraphTimeNodes[i], &params
        );
        if (err != cudaSuccess) { setLastError("gas graph time update", err); return 1; }
    }
    const cudaError_t err = cudaGraphLaunch(s->gasGraphExec, nullptr);
    if (err != cudaSuccess) { setLastError("gas graph launch", err); return 1; }
    return 0;
}

// Shared source composition: chemistry always uses explicit stage volumes.
// Audits are copied by the storage policy before the next half overwrites them.
template<class TrialPolicy,class GasHostState>
int prepareGasTrialTransport(GasHostState* s,const GasHostPolicy::Time dt)
{
    if constexpr (ugkwp::GasStateTraits<GasHostState>::speciesCount > 0)
    {
        {
            const int block=s->fixedCellBlockThreads,grid=(s->nCells+block-1)/block;
            using GasDevice=typename std::remove_pointer<decltype(s->deviceState)>::type;
            if constexpr(ugkwp::GasSstAuditCapability<GasDevice>::value)
            {
                prepareGasSstAuditKernel<<<grid,block,0,s->gasCaptureStream>>>(s->deviceState);
                const auto error=cudaGetLastError();
                if(error!=cudaSuccess){setLastError("prepare SST inventory audit",error);return 1;}
            }
            const bool chemistry=s->gasSpecies.mode==ugkwp::GasMode::MixtureChemistry;
            if constexpr(ugkwp::GasGeometryCapability<GasHostState>::value)
                if(ugkwp::gasMovingGeometry(*s))
                {
                    if(!ugkwp::validateGasGeometryTimeIntegrator(true,s->hostGasTimeIntegrator))
                    {setLastErrorText("moving gas geometry requires certified Euler stages");return 1;}
                    validateGasStageGeometryKernel<<<grid,block,0,s->gasCaptureStream>>>(s->deviceState,dt);
                    const auto error=cudaGetLastError();
                    if(error!=cudaSuccess){setLastError("gas stage geometry preflight",error);return 1;}
                    if(TrialPolicy::validate(s)!=0)return 1;
                }
            if(chemistry)
            {
                advanceGasChemistryKernel<<<grid,block,0,s->gasCaptureStream>>>
                    (s->deviceState,dt*GasHostPolicy::Time(0.5),TrialPolicy::stageVolumes(s,false));
                const auto error=cudaGetLastError();
                if(error!=cudaSuccess){setLastError("pre-transport gas chemistry",error);return 1;}
            }
            recoverGasPrimitivesKernel<<<grid,block,0,s->gasCaptureStream>>>(s->deviceState);
            const auto recoveryError=cudaGetLastError();
            if(recoveryError!=cudaSuccess){setLastError("pre-transport mixture recovery",recoveryError);return 1;}
            if(TrialPolicy::validate(s)!=0)return 1;
            if(chemistry && TrialPolicy::captureChemistryAudit(s,false)!=0)return 1;
            if constexpr(ugkwp::GasStateTraits<GasDevice>::speciesCount>0)
                if(s->hostTurbulenceModel!=0)
                {
                    if(s->hostTurbulenceModel==3)
                    {
                        recoverSstPrimitivesKernel<<<grid,block,0,s->gasCaptureStream>>>(s->deviceState);
                        const auto error=cudaGetLastError();
                        if(error!=cudaSuccess){setLastError("pre-transport SST recovery",error);return 1;}
                    }
                    computeGasPrimitiveGradientsKernel<<<grid,block,0,s->gasCaptureStream>>>(s->deviceState,true);
                    auto error=cudaGetLastError();
                    if(error!=cudaSuccess){setLastError("pre-transport gas gradients",error);return 1;}
                    if(s->hostTurbulenceModel==3)
                    {
                        computeGasGradientLimiterKernel<<<grid,block,0,s->gasCaptureStream>>>(s->deviceState);
                        error=cudaGetLastError();
                        if(error!=cudaSuccess){setLastError("pre-transport gas limiter",error);return 1;}
                    }
                    computeGasEddyViscosityKernel<<<grid,block,0,s->gasCaptureStream>>>(s->deviceState);
                    error=cudaGetLastError();
                    if(error!=cudaSuccess){setLastError("pre-transport eddy viscosity",error);return 1;}
                }
            const int faceBlock=s->fixedFaceBlockThreads,faceGrid=(s->nFaces+faceBlock-1)/faceBlock;
            if(faceGrid>0)
            {
                computeGasCourantFieldKernel<<<faceGrid,faceBlock,0,s->gasCaptureStream>>>(s->deviceState,dt);
                const auto error=cudaGetLastError();
                if(error!=cudaSuccess){setLastError("post-chemistry wave speed",error);return 1;}
            }
            // SST inletOutlet gradients and compression bounds consume the
            // fresh Riemann mass predictor, never prior-stage flux scratch.
            if constexpr(ugkwp::GasStateTraits<GasDevice>::speciesCount>0)
                if(s->hostTurbulenceModel==3)
                {
                    computeSstGradientsKernel<<<grid,block,0,s->gasCaptureStream>>>(s->deviceState);
                    auto error=cudaGetLastError();
                    if(error!=cudaSuccess){setLastError("pre-transport SST gradients",error);return 1;}
                    computeSstStabilityNumberKernel<<<grid,block,0,s->gasCaptureStream>>>(s->deviceState,dt,TrialPolicy::targetMaxCo(s));
                    error=cudaGetLastError();
                    if(error!=cudaSuccess){setLastError("pre-transport SST stability",error);return 1;}
                }
            computeGasConvectiveCourantByCellKernel<<<grid,block,0,s->gasCaptureStream>>>(s->deviceState,dt);
            auto error=cudaGetLastError();
            if(error!=cudaSuccess){setLastError("post-chemistry convective bound",error);return 1;}
            computeGasDiffusionNumberKernel<<<grid,block,0,s->gasCaptureStream>>>
                (s->deviceState,dt,TrialPolicy::targetMaxCo(s));
            error=cudaGetLastError();
            if(error!=cudaSuccess){setLastError("post-chemistry diffusion bound",error);return 1;}
            if(TrialPolicy::validate(s)!=0 || TrialPolicy::validateTimeStep(s,dt)!=0)return 1;
        }
    }
    return 0;
}

template<class TrialPolicy,class GasHostState>
int finaliseGasTrialSources(GasHostState* s,const GasHostPolicy::Time dt)
{
    if constexpr (ugkwp::GasStateTraits<GasHostState>::speciesCount > 0)
    {
        if(ugkwp::mixtureGasActive(*s))
        {
            const int block=s->fixedCellBlockThreads,grid=(s->nCells+block-1)/block;
            const bool chemistry=s->gasSpecies.mode==ugkwp::GasMode::MixtureChemistry;
            if(chemistry)
            {
                advanceGasChemistryKernel<<<grid,block,0,s->gasCaptureStream>>>
                    (s->deviceState,dt*GasHostPolicy::Time(0.5),TrialPolicy::stageVolumes(s,true));
                const auto error=cudaGetLastError();
                if(error!=cudaSuccess){setLastError("post-transport gas chemistry",error);return 1;}
            }
            recoverGasPrimitivesKernel<<<grid,block,0,s->gasCaptureStream>>>(s->deviceState);
            const auto error=cudaGetLastError();
            if(error!=cudaSuccess){setLastError("post-source mixture recovery",error);return 1;}
            if(TrialPolicy::validate(s)!=0)return 1;
            if(chemistry && TrialPolicy::captureChemistryAudit(s,true)!=0)return 1;
        }
    }
    return 0;
}

// TrialPolicy owns storage/publication only. Both production applications use
// this exact sequence; a failed cell/face rejects and restores the whole trial.
// begin must snapshot conserved/species state, boundary history and ledgers,
// then clear trial statuses. validate must synchronize and inspect all statuses.
// rollback restores accepted state and diagnostics; commit publishes once.
template<class TrialPolicy, class GasHostState>
int advanceGasTrial
(
    GasHostState* s,
    const GasHostPolicy::Time dt,
    const GasHostPolicy::Time simulationTime
)
{
    if (TrialPolicy::begin(s) != 0)
    { TrialPolicy::rollback(s); return 1; }
    if (prepareGasTrialTransport<TrialPolicy>(s,dt) != 0
        || advanceGasFluxStage(s, dt, simulationTime) != 0
        || TrialPolicy::applySources(s, dt) != 0
        || finaliseGasTrialSources<TrialPolicy>(s,dt) != 0
        || finaliseGasBoundaryStage(s, dt, simulationTime+dt) != 0
        || TrialPolicy::validate(s) != 0
        || TrialPolicy::commit(s) != 0)
    { TrialPolicy::rollback(s); return 1; }
    return 0;
}

template<class Time>
struct GasRequestedIntervalControls
{
    Time minimumSubstep = Time(1e-14);
    int maximumAttempts = 10000;
};

// Covers the entire requested interval or restores its initial accepted state.
// The application clock is never advanced here. Policy snapshots must include
// source budgets and diagnostics at BOTH the trial and requested-interval levels.
template<class TrialPolicy, class GasHostState>
int advanceGasRequestedInterval
(
    GasHostState* s,
    const GasHostPolicy::Time requestedDt,
    const GasHostPolicy::Time simulationTime,
    const GasRequestedIntervalControls<GasHostPolicy::Time>& controls
)
{
    using Time = GasHostPolicy::Time;
    if (!(requestedDt > Time(0)) || !(controls.minimumSubstep > Time(0))
        || controls.maximumAttempts <= 0 || simulationTime+requestedDt==simulationTime)
        return 1;
    if (TrialPolicy::beginInterval(s) != 0)
    { TrialPolicy::rollbackInterval(s); return 1; }
    Time elapsed = Time(0), substep = requestedDt;
    int attempts = 0;
    while (elapsed < requestedDt)
    {
        const Time remaining = requestedDt-elapsed;
        if (substep > remaining) substep=remaining;
        if (++attempts > controls.maximumAttempts || elapsed+substep==elapsed
            || !(substep > Time(0)))
        { TrialPolicy::rollbackInterval(s); return 1; }
        if (advanceGasTrial<TrialPolicy>(s,substep,simulationTime+elapsed)==0)
        {
            elapsed += substep;
            if (elapsed < requestedDt) substep*=Time(2);
        }
        else
        {
            substep*=Time(0.5);
            if (!TrialPolicy::retryable(s) || substep < controls.minimumSubstep)
            { TrialPolicy::rollbackInterval(s); return 1; }
        }
    }
    if (TrialPolicy::commitInterval(s)!=0)
    { TrialPolicy::rollbackInterval(s); return 1; }
    return 0;
}
