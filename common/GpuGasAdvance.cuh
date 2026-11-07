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
int advanceGasEulerSubstage(GasHostState* s, const GasHostPolicy::Time dt, const GasHostPolicy::Time ledgerDt, const GasHostPolicy::Time requestedTime, const bool nativeRefinement=false, const bool primitivesReady=false)
{
    const GasHostPolicy::Time stageTime = ugkwp::gasStageTime(*s, requestedTime);
    const int cellBlock = s->fixedCellBlockThreads;
    const int faceBlock = s->fixedFaceBlockThreads;
    const int cellGrid = (s->nCells + cellBlock - 1)/cellBlock;
    const int faceGrid = (s->nFaces + faceBlock - 1)/faceBlock;
    const int boundaryGrid = (s->nFaces - s->nInternalFaces + faceBlock - 1)/faceBlock;
    cudaError_t err = cudaSuccess;
    const bool preparedBoundaryLayer=ugkwp::gasBoundaryLayerPreparationReusable(*s,dt,stageTime);

    // Flux-stage callers already recovered this unchanged conservative state.
    // Direct substage callers (including steady residual evaluation) still recover.
    if (!primitivesReady)
    {
        recoverGasPrimitivesKernel<<<cellGrid, cellBlock, 0, s->gasCaptureStream>>>(s->deviceState,nativeRefinement);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("recoverGasPrimitivesKernel gas stage", err);
            return 1;
        }
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
        return advanceGasEulerSubstage(s, dt, dt, simulationTime, nativeRefinement, true);
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
    if (advanceGasEulerSubstage(s, dt, firstLedgerDt, simulationTime, nativeRefinement, true) != 0)
    {
        return 1;
    }
    if (advanceGasEulerSubstage(s, dt, firstLedgerDt, simulationTime+dt, nativeRefinement, true) != 0)
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
    if (advanceGasEulerSubstage(s, dt, GasHostPolicy::finalLedgerDt(dt), simulationTime+dt*GasHostPolicy::Time(0.5), nativeRefinement, true) != 0)
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

template<class State>
__global__ void finishResidentGasIntervalKernel(State* state)
{
    if(!blockIdx.x&&!threadIdx.x)state->gasResidentInterval.active=false;
}

template<class Policy,class State,class Time>
auto prepareResidentGasGraphCapture(State* state,Time dt,Time time,bool prepared,int)
    -> decltype(Policy::prepareGraphCapture(state,dt,time,prepared),void())
{Policy::prepareGraphCapture(state,dt,time,prepared);}
template<class Policy,class State,class Time>
void prepareResidentGasGraphCapture(State*,Time,Time,bool,long){}
template<class Policy,class State,class Time>
auto publishResidentGasGraphReplay(State* state,Time dt,Time time,int)
    -> decltype(Policy::publishGraphReplay(state,dt,time),void())
{Policy::publishGraphReplay(state,dt,time);}
template<class Policy,class State,class Time>
void publishResidentGasGraphReplay(State*,Time,Time,long){}

template<class GasHostState>
int advanceResidentGasGraph(GasHostState* s,const GasHostPolicy::Time dt,
    const GasHostPolicy::Time requestedTime,const bool finalBoundary)
{
    const auto simulationTime=ugkwp::gasStageTime(*s,requestedTime);
    const bool prepared=ugkwp::gasBoundaryLayerPreparationReusable(*s,dt,simulationTime);
    ugkwp::GasResidentInterval interval;
    interval.dt=dt;interval.begin=simulationTime;
    interval.end=simulationTime+dt;interval.middle=simulationTime+dt*GasHostPolicy::Time(0.5);
    interval.active=true;
    if constexpr(ugkwp::HasGasFrozenBoundaryTime<GasHostState>::value)
        interval.frozen=s->gasSteadyBoundaryFrozen;
    else if constexpr(ugkwp::HasGasSteadyTime<GasHostState>::value)
        interval.frozen=s->gasSteadyMode;
    const auto executionStream=s->gasCaptureStream;
    if(!s->gasGraphExec || s->gasGraphDt!=s->gasGraphDt
        ||s->gasGraphIncludesBoundary!=finalBoundary||s->gasGraphPreparedBoundary!=prepared
        ||s->gasGraphFrozenBoundary!=interval.frozen)
    {
        auto error=cudaSuccess;
        if(s->gasGraphExec){error=cudaGraphExecDestroy(s->gasGraphExec);s->gasGraphExec=nullptr;}
        if(error==cudaSuccess&&s->gasGraph){error=cudaGraphDestroy(s->gasGraph);s->gasGraph=nullptr;}
        if(error!=cudaSuccess){setLastError("release resident gas graph",error);return 1;}
        s->gasGraphIntervalNode=nullptr;s->gasGraphIntervalSubmitted=false;
        // The clock has its own owner: particle capacity/geometry updates may
        // republish the device state header between otherwise identical steps.
        if(!s->gasGraphClock){
            error=cudaMalloc(reinterpret_cast<void**>(&s->gasGraphClock),sizeof(ugkwp::GasResidentInterval));
            if(error!=cudaSuccess){setLastError("allocate resident gas clock",error);return 1;}
        }
        cudaStream_t capture=nullptr;
        error=cudaStreamCreateWithFlags(&capture,cudaStreamNonBlocking);
        if(error!=cudaSuccess){setLastError("create resident gas capture stream",error);return 1;}
        error=cudaStreamBeginCapture(capture,cudaStreamCaptureModeThreadLocal);
        if(error!=cudaSuccess){cudaStreamDestroy(capture);setLastError("capture resident gas step",error);return 1;}
        s->gasCaptureStream=capture;
        // Capture relative stage times and RK weights once. The first node
        // supplies the physical FP64 interval on every replay.
        if constexpr(ugkwp::GasBoundaryLayerCapability<GasHostState>::value)
            if(prepared){s->gasBoundaryLayer.preparedInterval=1;
                s->gasBoundaryLayer.preparedTime=ugkwp::gasStageTime(*s,0.0);}
        prepareResidentGasGraphCapture<GasHostPolicy>(s,dt,simulationTime,prepared,0);
        ugkwp::setResidentGasIntervalKernel<<<1,1,0,capture>>>(s->deviceState,s->gasGraphClock,interval,false);
        int status=advanceGasFluxStage(s,GasHostPolicy::Time(1),GasHostPolicy::Time(0));
        if(!status&&finalBoundary)status=finaliseGasBoundaryStage(s,GasHostPolicy::Time(1),GasHostPolicy::Time(1));
        finishResidentGasIntervalKernel<<<1,1,0,capture>>>(s->deviceState);
        error=cudaStreamEndCapture(capture,&s->gasGraph);
        const auto destroyError=cudaStreamDestroy(capture);
        s->gasCaptureStream=executionStream;
        if(status)return status;
        if(error!=cudaSuccess){setLastError("finish resident gas capture",error);return 1;}
        if(destroyError!=cudaSuccess){setLastError("release resident gas capture stream",destroyError);return 1;}
        size_t count=0;
        error=cudaGraphGetNodes(s->gasGraph,nullptr,&count);
        if(error!=cudaSuccess){setLastError("resident gas node count",error);return 1;}
        std::vector<cudaGraphNode_t> nodes(count);
        error=cudaGraphGetNodes(s->gasGraph,nodes.data(),&count);
        if(error!=cudaSuccess){setLastError("resident gas nodes",error);return 1;}
        using Device=typename std::remove_pointer<decltype(s->deviceState)>::type;
        for(auto node:nodes){
            cudaGraphNodeType type;error=cudaGraphNodeGetType(node,&type);
            if(error!=cudaSuccess){setLastError("resident gas node type",error);return 1;}
            if(type!=cudaGraphNodeTypeKernel)continue;
            cudaKernelNodeParams params{};error=cudaGraphKernelNodeGetParams(node,&params);
            if(error!=cudaSuccess){setLastError("resident gas node parameters",error);return 1;}
            if(params.func==reinterpret_cast<void*>(ugkwp::setResidentGasIntervalKernel<Device>)){
                s->gasGraphIntervalNode=node;params.kernelParams=nullptr;params.extra=nullptr;
                s->gasGraphIntervalParams=params;
            }
        }
        if(!s->gasGraphIntervalNode){setLastErrorText("resident gas interval node unavailable");return 1;}
        error=cudaGraphInstantiate(&s->gasGraphExec,s->gasGraph,nullptr,nullptr,0);
        if(error!=cudaSuccess){setLastError("instantiate resident gas graph",error);return 1;}
        s->gasGraphIncludesBoundary=finalBoundary;s->gasGraphPreparedBoundary=prepared;
        s->gasGraphFrozenBoundary=interval.frozen;s->gasGraphDt=dt;
        // Stream capture already supplied this exact interval to the first
        // kernel. Keep the same node data while the requested interval is equal.
        s->gasGraphLastInterval=interval;s->gasGraphIntervalSubmitted=true;
        s->gasGraphAdvancesInterval=false;
    }
    auto error=cudaSuccess;
    bool advance=s->gasGraphIntervalSubmitted
        &&ugkwp::canAdvanceGasResidentInterval(s->gasGraphLastInterval,interval);
    if(!s->gasGraphIntervalSubmitted||advance!=s->gasGraphAdvancesInterval
        ||(!advance&&!ugkwp::sameGasResidentInterval(interval,s->gasGraphLastInterval))){
        auto* device=s->deviceState;auto* clock=s->gasGraphClock;
        void* args[]={&device,&clock,&interval,&advance};
        auto params=s->gasGraphIntervalParams;params.kernelParams=args;
        error=cudaGraphExecKernelNodeSetParams(s->gasGraphExec,s->gasGraphIntervalNode,&params);
        if(error==cudaSuccess)s->gasGraphAdvancesInterval=advance;
    }
    if(error==cudaSuccess)error=cudaGraphLaunch(s->gasGraphExec,executionStream);
    if(error==cudaSuccess){s->gasGraphLastInterval=interval;s->gasGraphIntervalSubmitted=true;}
    if(error!=cudaSuccess){setLastError("replay resident gas step",error);return 1;}
    if constexpr(ugkwp::GasBoundaryLayerCapability<GasHostState>::value){
        s->gasBoundaryLayer.preparedFirstStage=false;
        s->gasBoundaryLayerStageTime=ugkwp::gasStageTime(*s,
            simulationTime+(s->hostGasTimeIntegrator==1?0:s->hostGasTimeIntegrator==2?dt:dt*GasHostPolicy::Time(0.5)));
    }
    publishResidentGasGraphReplay<GasHostPolicy>(s,dt,simulationTime,0);
    return 0;
}
template<class GasHostState>
int advancePureGasGraph(GasHostState* s,const GasHostPolicy::Time dt,const GasHostPolicy::Time time)
{return advanceResidentGasGraph(s,dt,time,true);}

template<class State,class=void>struct HasResidentGasGraph:std::false_type{};
template<class State>struct HasResidentGasGraph<State,std::void_t<decltype(std::declval<State>().gasGraphExec),
    decltype(std::declval<State>().deviceState->gasResidentInterval),
    decltype(std::declval<State>().gasGraphIntervalNode),
    decltype(std::declval<State>().gasGraphLastInterval),
    decltype(std::declval<State>().gasGraphIntervalSubmitted),
    decltype(std::declval<State>().gasGraphAdvancesInterval),
    decltype(std::declval<State>().gasGraphClock)>>:std::true_type{};
template<class GasHostState>
int advanceGasFluxScheduled(GasHostState* s,const GasHostPolicy::Time dt,
    const GasHostPolicy::Time time,const bool nativeRefinement=false)
{
#ifndef UGKP_DEVELOPMENT_PROBES
    if constexpr(HasResidentGasGraph<GasHostState>::value){
        if(s->gasGraphMode<0){const char* value=std::getenv("UGKP_GAS_GRAPH");
            s->gasGraphMode=!value||std::strcmp(value,"1")==0;}
        if(s->gasGraphMode&&!nativeRefinement)return advanceResidentGasGraph(s,dt,time,false);
    }
#endif
    return advanceGasFluxStage(s,dt,time,nativeRefinement);
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
        || advanceGasFluxScheduled(s, dt, simulationTime,nativeRefinement) != 0
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
