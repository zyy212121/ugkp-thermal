#pragma once
#include <type_traits>
#include "gasTransport/GasPseudoTime.H"
#include "gasTransport/GasStateView.H"
#include "gasTransport/GasBoundaryLayerHost.H"
#include "operators/advanceGasChemistryKernel.cuh"

// Shared Euler/RK, boundary-finalisation and graph-capture host protocol.
// Requires a gas host state, GasHostPolicy, all gas kernels and error helpers.
// The host state owns launch/graph resources and a deviceState pointer. That
// pointer may name a gas-only view; no particle storage is required here.
// GasHostPolicy owns only time/weight types and the optional wall-energy ledger;
// launches, streams, error checks and their ordering are owned here.

template<class GasHostState>
int advanceGasEulerSubstage(GasHostState* s, const GasHostPolicy::Time dt, const GasHostPolicy::Time ledgerDt, const GasHostPolicy::Time requestedTime, const bool nativeRefinement=false)
{
    const GasHostPolicy::Time stageTime = ugkwp::gasStageTime(*s, requestedTime);
    const int cellBlock = s->fixedCellBlockThreads;
    const int faceBlock = s->fixedFaceBlockThreads;
    const int cellGrid = (s->nCells + cellBlock - 1)/cellBlock;
    const int faceGrid = (s->nFaces + faceBlock - 1)/faceBlock;
    const int boundaryGrid = (s->nFaces - s->nInternalFaces + faceBlock - 1)/faceBlock;
    cudaError_t err = cudaSuccess;
    const bool preparedBoundaryLayer=ugkwp::gasBoundaryLayerPreparationReusable(*s,dt,stageTime);

    recoverGasPrimitivesKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState,nativeRefinement);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("recoverGasPrimitivesKernel gas stage", err);
        return 1;
    }
    if (s->hostTurbulenceModel == 3)
    {
        recoverSstPrimitivesKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState,nativeRefinement);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("recoverSstPrimitivesKernel gas stage", err);
            return 1;
        }
    }
    if (boundaryGrid > 0 && !preparedBoundaryLayer)
    {
        updateRiemannBoundaryMirrorKernel<<<boundaryGrid, faceBlock, 0, s->gasCaptureStream>>>(s->deviceState,stageTime);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("updateRiemannBoundaryMirrorKernel gas stage", err);
            return 1;
        }
    }
    if(ugkwp::gasBoundaryLayerEnabled(*s) && !preparedBoundaryLayer)
    {
        computeGasPrimitiveGradientsKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState,nativeRefinement);
        err=cudaGetLastError();
        if(err!=cudaSuccess){setLastError("wall stage primitive gradients",err);return 1;}
    }
    if(ugkwp::prepareGasBoundaryLayerStage<GasHostPolicy>(s,dt,stageTime,false)!=0)
    {setLastErrorText("boundaryLayer stage preparation failed");return 1;}
    if (s->hostTurbulenceModel == 3 && !preparedBoundaryLayer)
    {
        applySstWallFunctionStateKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState,nativeRefinement);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("applySstWallFunctionStateKernel gas stage", err);
            return 1;
        }
    }
    if (s->hostGasFluxScheme == 7 && !ugkwp::mixtureGasActive(*s))
    {
        computeGasHllcAdcSensorKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("computeGasHllcAdcSensorKernel launch", err);
            return 1;
        }
    }
    if(!ugkwp::gasBoundaryLayerEnabled(*s))
    {
        computeGasPrimitiveGradientsKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState,nativeRefinement);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("computeGasPrimitiveGradientsKernel launch", err);
            return 1;
        }
    }
    computeGasGradientLimiterKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState,nativeRefinement);
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
            dt,nativeRefinement);
    }
    else
    {
        computeGasInternalFaceFluxKernel<true><<<faceGrid, faceBlock, 0, s->gasCaptureStream>>>
        (
            s->deviceState,
            dt,nativeRefinement);
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
            s->deviceState,nativeRefinement);
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
        s->deviceState,nativeRefinement);
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
            s->deviceState,nativeRefinement);
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
        if(GasHostPolicy::applySstFaceSources(s,dt)!=0)return 1;
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
            dt,nativeRefinement);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("applySstFluxAndSourceKernel launch", err);
            return 1;
        }
    }
    applyGasFluxDivergenceByCellKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState, dt, nativeRefinement);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("applyGasFluxDivergenceByCellKernel launch", err);
        return 1;
    }
    recoverGasPrimitivesKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState,nativeRefinement);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("recoverGasPrimitivesKernel post-gas launch", err);
        return 1;
    }
    if (s->hostTurbulenceModel == 3)
    {
        recoverSstPrimitivesKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState,nativeRefinement);
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
    const double initialWeight,
    const double stageWeight,
    const bool nativeRefinement=false
)
{
    const int block = s->fixedCellBlockThreads;
    const int cellGrid = (s->nCells + block - 1)/block;
    blendGasConservativeStateKernel<<<cellGrid, block, 0, s->gasCaptureStream>>>
    (
        s->deviceState,
        initialWeight,
        stageWeight, nativeRefinement
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("blendGasConservativeStateKernel launch", err);
        return 1;
    }
    recoverGasPrimitivesKernel<<<cellGrid, block, 0, s->gasCaptureStream>>>(s->deviceState,nativeRefinement);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("recoverGasPrimitivesKernel after RK blend", err);
        return 1;
    }
    if (s->hostTurbulenceModel == 3)
    {
        recoverSstPrimitivesKernel<<<cellGrid, block, 0, s->gasCaptureStream>>>(s->deviceState,nativeRefinement);
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
int advanceGasFluxStage(GasHostState* s, const GasHostPolicy::Time dt, const GasHostPolicy::Time requestedTime, const bool nativeRefinement=false)
{
    const GasHostPolicy::Time simulationTime = ugkwp::gasStageTime(*s, requestedTime);

    const int preparationCellBlock = s->fixedCellBlockThreads;
    const int preparationFaceBlock = s->fixedFaceBlockThreads;
    const int preparationCellGrid =
        (s->nCells + preparationCellBlock - 1)/preparationCellBlock;
    const int preparationFaceGrid =
        (s->nFaces - s->nInternalFaces + preparationFaceBlock - 1)/preparationFaceBlock;
    recoverGasPrimitivesKernel<<<preparationCellGrid, preparationCellBlock, 0, s->gasCaptureStream>>>
    (
        s->deviceState,nativeRefinement);
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
    if (preparationFaceGrid > 0
        && !ugkwp::gasBoundaryLayerPreparationReusable(*s,dt,simulationTime))
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
        return advanceGasEulerSubstage(s, dt, dt, simulationTime, nativeRefinement);
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
    if (advanceGasEulerSubstage(s, dt, firstLedgerDt, simulationTime, nativeRefinement) != 0)
    {
        return 1;
    }
    if (advanceGasEulerSubstage(s, dt, firstLedgerDt, simulationTime+dt, nativeRefinement) != 0)
    {
        return 1;
    }

    if (s->hostGasTimeIntegrator == 2)
    {

        return blendGasRungeKuttaStage(s, static_cast<double>(0.5), static_cast<double>(0.5), nativeRefinement);
    }
    if (s->hostGasTimeIntegrator != 3)
    {
        setLastErrorText("invalid resident gas time-integrator selector");
        return 1;
    }

    if (blendGasRungeKuttaStage(s, static_cast<double>(0.75), static_cast<double>(0.25), nativeRefinement) != 0)
    {
        return 1;
    }
    if (advanceGasEulerSubstage(s, dt, GasHostPolicy::finalLedgerDt(dt), simulationTime+dt*GasHostPolicy::Time(0.5), nativeRefinement) != 0)
    {
        return 1;
    }
    return blendGasRungeKuttaStage
    (
        s,
        static_cast<double>(1.0)/static_cast<double>(3.0),
        static_cast<double>(2.0)/static_cast<double>(3.0),
        nativeRefinement
    );
}

template<class GasHostState>
int finaliseGasBoundaryStage(GasHostState* s, const GasHostPolicy::Time dt, const GasHostPolicy::Time requestedTime)
{
    const GasHostPolicy::Time simulationTime = ugkwp::gasStageTime(*s, requestedTime);
    const int block = s->fixedFaceBlockThreads;
    const int allFaceGrid = (s->nFaces - s->nInternalFaces + block - 1)/block;
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
        s->deviceState,simulationTime
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
int advancePureGasGraph(GasHostState* s, const GasHostPolicy::Time dt, const GasHostPolicy::Time requestedTime)
{
    const GasHostPolicy::Time simulationTime = ugkwp::gasStageTime(*s, requestedTime);
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
        s->gasGraphTimeOffsets.clear();
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
        // Graph transport receives the interval start; final publication uses its end.
        if (status == 0) status = finaliseGasBoundaryStage(s, dt, simulationTime+dt);
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
            if (params.func == reinterpret_cast<void*>(updateLegacyGasBoundaryMirrorKernel<GasDeviceState>)
                ||params.func == reinterpret_cast<void*>(updateRiemannBoundaryMirrorKernel<GasDeviceState>))
            {
                s->gasGraphTimeNodes.push_back(node);
                s->gasGraphTimeOffsets.push_back(*static_cast<GasHostPolicy::Time*>(params.kernelParams[1])-simulationTime);
                params.kernelParams = nullptr;
                params.extra = nullptr;
                s->gasGraphTimeParams.push_back(params);
            }
        }
        if (s->nFaces > s->nInternalFaces && s->gasGraphTimeNodes.size() < 2)
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
        currentTime=simulationTime+s->gasGraphTimeOffsets[i];
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



template<class Policy,class State>
auto validateGasTrialState(State* s,bool nativeRefinement,int)
    -> decltype(Policy::validate(s,nativeRefinement))
{return Policy::validate(s,nativeRefinement);}
template<class Policy,class State>
int validateGasTrialState(State* s,bool,long)
{return Policy::validate(s);}

template<class Policy,class State,class Time>
auto applyGasTrialSources(State* s,Time dt,bool nativeRefinement,int)
    -> decltype(Policy::applySources(s,dt,nativeRefinement))
{return Policy::applySources(s,dt,nativeRefinement);}
template<class Policy,class State,class Time>
int applyGasTrialSources(State* s,Time dt,bool,long)
{return Policy::applySources(s,dt);}


// Shared source composition: chemistry always uses explicit stage volumes.
template<class TrialPolicy,class GasHostState>
int prepareGasTrialTransport(GasHostState* s,const GasHostPolicy::Time dt,
    const GasHostPolicy::Time requestedTime=GasHostPolicy::Time(0),
    const bool nativeRefinement=false)
{
    const GasHostPolicy::Time simulationTime = ugkwp::gasStageTime(*s, requestedTime);
    if constexpr (ugkwp::GasStateTraits<GasHostState>::speciesCount > 0)
    {
        {
            const int block=s->fixedCellBlockThreads,grid=(s->nCells+block-1)/block;
            const bool chemistry=s->gasSpecies.mode==ugkwp::GasMode::MixtureChemistry;
            if(chemistry)
            {
                advanceGasChemistryKernel<<<grid,block,0,s->gasCaptureStream>>>
                    (s->deviceState,dt*GasHostPolicy::Time(0.5),TrialPolicy::stageVolumes(s,false),nativeRefinement);
                const auto error=cudaGetLastError();
                if(error!=cudaSuccess){setLastError("pre-transport gas chemistry",error);return 1;}
            }
            recoverGasPrimitivesKernel<<<grid,block,0,s->gasCaptureStream>>>(s->deviceState,nativeRefinement);
            const auto recoveryError=cudaGetLastError();
            if(recoveryError!=cudaSuccess){setLastError("pre-transport mixture recovery",recoveryError);return 1;}
            if(ugkwp::gasBoundaryLayerEnabled(*s))
            {
                // The first chemistry half-step has completed. Refresh only the
                // selected family's current boundary input and prepare once;
                // the first Euler stage consumes this exact state/phase cache.
                if(s->hostTurbulenceModel==3)
                {
                    recoverSstPrimitivesKernel<<<grid,block,0,s->gasCaptureStream>>>(s->deviceState,nativeRefinement);
                    const auto error=cudaGetLastError();
                    if(error!=cudaSuccess){setLastError("wall preflight SST recovery",error);return 1;}
                }
                const int faceBlock=s->fixedFaceBlockThreads,faceGrid=(s->nFaces-s->nInternalFaces+faceBlock-1)/faceBlock;
                if(faceGrid>0)
                {
                    updateLegacyGasBoundaryMirrorKernel<<<faceGrid,faceBlock,0,s->gasCaptureStream>>>(s->deviceState,simulationTime);
                    auto error=cudaGetLastError();
                    if(error!=cudaSuccess){setLastError("wall preflight boundary time",error);return 1;}
                    updateRiemannBoundaryMirrorKernel<<<faceGrid,faceBlock,0,s->gasCaptureStream>>>(s->deviceState,simulationTime);
                    error=cudaGetLastError();
                    if(error!=cudaSuccess){setLastError("wall preflight boundary state",error);return 1;}
                }
                computeGasPrimitiveGradientsKernel<<<grid,block,0,s->gasCaptureStream>>>(s->deviceState,nativeRefinement);
                const auto gradientError=cudaGetLastError();
                if(gradientError!=cudaSuccess){setLastError("wall preflight primitive gradients",gradientError);return 1;}
                if(ugkwp::prepareGasBoundaryLayerStage<GasHostPolicy>(s,dt,simulationTime,true)!=0)
                {setLastErrorText("boundaryLayer preflight preparation failed");return 1;}
                if(s->hostTurbulenceModel==3)
                {
                    applySstWallFunctionStateKernel<<<grid,block,0,s->gasCaptureStream>>>(s->deviceState,nativeRefinement);
                    const auto error=cudaGetLastError();
                    if(error!=cudaSuccess){setLastError("wall preflight owner omega",error);return 1;}
                }
            }
            if(validateGasTrialState<TrialPolicy>(s,nativeRefinement,0)!=0)return 1;


        }
    }
    return 0;
}

template<class TrialPolicy,class GasHostState>
int finaliseGasTrialSources(GasHostState* s,const GasHostPolicy::Time dt,
    const bool nativeRefinement=false)
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
                    (s->deviceState,dt*GasHostPolicy::Time(0.5),TrialPolicy::stageVolumes(s,true),nativeRefinement);
                const auto error=cudaGetLastError();
                if(error!=cudaSuccess){setLastError("post-transport gas chemistry",error);return 1;}
            }
            recoverGasPrimitivesKernel<<<grid,block,0,s->gasCaptureStream>>>(s->deviceState,nativeRefinement);
            const auto error=cudaGetLastError();
            if(error!=cudaSuccess){setLastError("post-source mixture recovery",error);return 1;}
            if(validateGasTrialState<TrialPolicy>(s,nativeRefinement,0)!=0)return 1;
        }
    }
    return 0;
}




template<class TrialPolicy, class GasHostState>
int advanceGasTrial
(
    GasHostState* s,
    const GasHostPolicy::Time dt,
    const GasHostPolicy::Time simulationTime,
    const bool nativeRefinement=false
)
{
    ugkwp::invalidateGasBoundaryLayerPreparation(*s);
    if (TrialPolicy::begin(s) != 0)
    { ugkwp::invalidateGasBoundaryLayerPreparation(*s); return 1; }
    if (prepareGasTrialTransport<TrialPolicy>(s,dt,simulationTime,nativeRefinement) != 0
        || advanceGasFluxStage(s, dt, simulationTime,nativeRefinement) != 0
        || (!nativeRefinement && validateGasTrialState<TrialPolicy>(s,nativeRefinement,0) != 0)
        || applyGasTrialSources<TrialPolicy>(s,dt,nativeRefinement,0) != 0
        || finaliseGasTrialSources<TrialPolicy>(s,dt,nativeRefinement) != 0
        || finaliseGasBoundaryStage(s, dt, simulationTime+dt) != 0
        || validateGasTrialState<TrialPolicy>(s,nativeRefinement,0) != 0
        || TrialPolicy::commit(s) != 0)
    { ugkwp::invalidateGasBoundaryLayerPreparation(*s); return 1; }
    return 0;
}

template<class Time>
struct GasRequestedIntervalControls
{
    // Compatibility token: the requested interval has no retry budget.
};




template<class TrialPolicy, class GasHostState>
int advanceGasRequestedInterval
(
    GasHostState* s,
    const GasHostPolicy::Time requestedDt,
    const GasHostPolicy::Time simulationTime,
    const GasRequestedIntervalControls<GasHostPolicy::Time>& controls
)
{
    (void)controls;
    if (TrialPolicy::beginInterval(s) != 0) return 1;
    if (advanceGasTrial<TrialPolicy>(s,requestedDt,simulationTime,false) != 0) return 1;
    return TrialPolicy::commitInterval(s);
}
