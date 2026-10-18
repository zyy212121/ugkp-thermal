#pragma once
#include <cmath>



struct GasCourantMaxima
{
    double convective=0, diffusion=0, sst=0, maximum=0;
};
// Adapters with backend-specific status handling may retain scalar reductions.
// The native gas backend supplies the joint overload, with one host readback.
template<class Reduction,class Real> auto reduceGasCourantFields(Reduction reduce,
 const Real* convective,const Real* diffusion,const Real* sst,int count,cudaStream_t stream,
 GasCourantMaxima& result,int)->decltype(reduce(convective,diffusion,sst,count,stream,result)){
    return reduce(convective,diffusion,sst,count,stream,result);
}
template<class Reduction,class Real> int reduceGasCourantFields(Reduction reduce,
 const Real* convective,const Real* diffusion,const Real* sst,int count,cudaStream_t stream,
 GasCourantMaxima& result,long){
    if(reduce(convective,count,stream,result.convective)||reduce(diffusion,count,stream,result.diffusion))return 1;
    return sst?reduce(sst,count,stream,result.sst):0;
}
template<class GasHostState,class Time,class Real,class MaximumReduction>
int computeGasCurrentCourant
(
    GasHostState* s, Time dt, Real targetMaxCo, Time scheduleTime,
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
    if (s->hostTurbulenceModel == 3)
    {
        recoverSstPrimitivesKernel<<<cellGrid, cellBlock, 0, stream>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("recoverSstPrimitivesKernel for gas Courant", err);
            return 1;
        }
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
            s->deviceState
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

    if (s->hostTurbulenceModel == 3)
    {
        applySstWallFunctionStateKernel<<<cellGrid, cellBlock, 0, stream>>>(s->deviceState, true);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError
            (
                "applySstWallFunctionStateKernel for gas Courant",
                err
            );
            return 1;
        }
    }

    computeGasPrimitiveGradientsKernel<<<cellGrid, cellBlock, 0, stream>>>(s->deviceState, true);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("computeGasPrimitiveGradientsKernel for gas Courant", err);
        return 1;
    }
    computeGasGradientLimiterKernel<<<cellGrid, cellBlock, 0, stream>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("computeGasGradientLimiterKernel for gas Courant", err);
        return 1;
    }
    computeGasEddyViscosityKernel<<<cellGrid, cellBlock, 0, stream>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("computeGasEddyViscosityKernel for gas Courant", err);
        return 1;
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
    if (s->hostTurbulenceModel == 3)
    {
        computeSstGradientsKernel<<<cellGrid, cellBlock, 0, stream>>>(s->deviceState);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("computeSstGradientsKernel for gas Courant", err);
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

    computeGasDiffusionNumberKernel<<<cellGrid, cellBlock, 0, stream>>>
    (
        s->deviceState,
        dt,
        targetMaxCo
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("computeGasDiffusionNumberKernel launch", err);
        return 1;
    }
    if (s->hostTurbulenceModel == 3)
    {
        computeSstStabilityNumberKernel<<<cellGrid, cellBlock, 0, stream>>>
        (
            s->deviceState,
            dt,
            targetMaxCo,
            false
        );
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            setLastError("computeSstStabilityNumberKernel launch", err);
            return 1;
        }
    }


    GasCourantMaxima computed;
    if(reduceGasCourantFields(maximumReduction,s->gasFluxPositivityScale,s->gasDiffusionNumber,
        s->hostTurbulenceModel==3?s->sstSourceNumber:nullptr,s->nCells,stream,computed,0))return 1;
    computed.maximum=std::fmax(computed.convective,computed.diffusion);
    if(s->hostTurbulenceModel==3)computed.maximum=std::fmax(computed.maximum,computed.sst);
    result=computed;
    return 0;
}
