#pragma once

// Shared Euler/RK, boundary-finalisation and graph-capture host protocol.
// Requires DeviceState, GasHostPolicy, all gas kernels and error helpers.
// GasHostPolicy owns only time/weight types and the optional wall-energy ledger;
// launches, streams, error checks and their ordering are owned here.
// RK fractions are converted before division to preserve FP32 weight semantics.
int advanceGasEulerSubstage(DeviceState* s, const GasHostPolicy::Time dt, const GasHostPolicy::Time ledgerDt)
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

int blendGasRungeKuttaStage
(
    DeviceState* s,
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

int advanceGasFluxStage(DeviceState* s, const GasHostPolicy::Time dt, const GasHostPolicy::Time simulationTime)
{

    const int preparationCellBlock = s->fixedCellBlockThreads;
    const int preparationFaceBlock = s->fixedFaceBlockThreads;
    const int preparationCellGrid =
        (s->nCells + preparationCellBlock - 1)/preparationCellBlock;
    const int preparationFaceGrid =
        (s->nFaces + preparationFaceBlock - 1)/preparationFaceBlock;
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
        return advanceGasEulerSubstage(s, dt, dt);
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
    if (advanceGasEulerSubstage(s, dt, firstLedgerDt) != 0)
    {
        return 1;
    }
    if (advanceGasEulerSubstage(s, dt, firstLedgerDt) != 0)
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
    if (advanceGasEulerSubstage(s, dt, GasHostPolicy::finalLedgerDt(dt)) != 0)
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

int finaliseGasBoundaryStage(DeviceState* s, const GasHostPolicy::Time dt, const GasHostPolicy::Time simulationTime)
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

int advancePureGasGraph(DeviceState* s, const GasHostPolicy::Time dt, const GasHostPolicy::Time simulationTime)
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
            if (params.func == reinterpret_cast<void*>(updateLegacyGasBoundaryMirrorKernel))
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
    DeviceState* device = s->deviceState;
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
