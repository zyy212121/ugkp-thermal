#pragma once

struct GasCourantMaxima
{
    double convective=0, maximum=0;
};
template<class GasHostState,class Time,class MaximumReduction>
int computeGasCurrentCourant
(
    GasHostState* s, Time dt, Time scheduleTime,
    int cellBlock, int faceBlock, cudaStream_t stream,
    GasCourantMaxima& result, MaximumReduction maximumReduction
)
{
    const int cellGrid = (s->nCells + cellBlock - 1)/cellBlock;
    const int faceGrid = (s->nFaces + faceBlock - 1)/faceBlock;
    const int allFaceGrid = (s->nFaces - s->nInternalFaces + faceBlock - 1)/faceBlock;

    recoverGasPrimitivesKernel<<<cellGrid, cellBlock, 0, stream>>>(s->deviceState);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("recoverGasPrimitivesKernel launch for gas Courant", err);
        return 1;
    }

    if (allFaceGrid > 0)
    {
        updateLegacyGasBoundaryMirrorKernel<<<allFaceGrid, faceBlock, 0, stream>>>
        (
            s->deviceState,
            scheduleTime
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError
            (
                "updateLegacyGasBoundaryMirrorKernel for gas Courant",
                err
            );
            return 1;
        }
        updateRiemannBoundaryMirrorKernel<<<allFaceGrid, faceBlock, 0, stream>>>
        (
            s->deviceState,scheduleTime
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError
            (
                "updateRiemannBoundaryMirrorKernel for gas Courant",
                err
            );
            return 1;
        }
    }

    if (faceGrid > 0)
    {
        computeGasCourantFieldKernel<<<faceGrid, faceBlock, 0, stream>>>(s->deviceState, dt);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("computeGasCourantFieldKernel launch", err);
            return 1;
        }
    }

    computeGasConvectiveCourantByCellKernel<<<cellGrid, cellBlock, 0, stream>>>
    (
        s->deviceState,
        dt
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError
        (
            "computeGasConvectiveCourantByCellKernel launch",
            err
        );
        return 1;
    }

    GasCourantMaxima computed;
    if(maximumReduction(s->gasFluxPositivityScale,s->nCells,stream,computed.convective))return 1;
    computed.maximum=computed.convective;
    result=computed;
    return 0;
}
