#pragma once
#include <type_traits>
#include "gasTransport/GasTransportPreparation.H"

template<class GasHostState>
int prepareGasTransportCoefficients(GasHostState* s,int cellBlock,cudaStream_t stream,
    bool nativeRefinement=false,bool gradientsReady=false,bool reconstructFaces=false,
    const ugkwp::GasTransportPreparation* preparation=nullptr)
{
    const int cellGrid=(s->nCells+cellBlock-1)/cellBlock;
    cudaError_t error=cudaSuccess;
    if(!gradientsReady){
        computeGasPrimitiveGradientsKernel<<<cellGrid,cellBlock,0,stream>>>(s->deviceState,nativeRefinement,preparation);
        error=cudaGetLastError();
        if(error!=cudaSuccess){setLastError("transport primitive gradients",error);return 1;}
    }
    if(reconstructFaces){
        if(s->hostTurbulenceModel==0){
            using CellState=std::remove_pointer_t<decltype(s->deviceState)>;
            computeGasGradientLimiterKernel<CellState,true><<<cellGrid,cellBlock,0,stream>>>(s->deviceState,nativeRefinement,preparation);
        }else{
            computeGasGradientLimiterKernel<<<cellGrid,cellBlock,0,stream>>>(s->deviceState,nativeRefinement,preparation);
        }
        error=cudaGetLastError();
        if(error!=cudaSuccess){setLastError("transport gradient limiter",error);return 1;}
    }
    if(s->hostTurbulenceModel!=0){
        computeGasEddyViscosityKernel<<<cellGrid,cellBlock,0,stream>>>(s->deviceState,preparation);
        error=cudaGetLastError();
        if(error!=cudaSuccess){setLastError("transport eddy viscosity",error);return 1;}
    }
    return 0;
}

template<class GasHostState>
int prepareGasSstTransportCoefficients(GasHostState* s,int cellBlock,cudaStream_t stream,
    const ugkwp::GasTransportPreparation* preparation=nullptr)
{
    const int cellGrid=(s->nCells+cellBlock-1)/cellBlock;
    cudaError_t error=cudaSuccess;
    if(s->hostTurbulenceModel==3){
        computeSstGradientsKernel<<<cellGrid,cellBlock,0,stream>>>(s->deviceState,preparation);
        error=cudaGetLastError();
        if(error!=cudaSuccess){setLastError("transport SST gradients",error);return 1;}
    }
    return 0;
}

template<class GasState>
__global__ void prepareGasSstBoundaryMassFluxKernel(GasState* sp)
{
    if(!ugkp::diagnostics::blockMayProceed(sp->diagnostics))return;
    auto& s=*sp;
    const int f=s.nInternalFaces+blockIdx.x*blockDim.x+threadIdx.x;
    if(f>=s.nFaces || s.riemannBoundaryKind[f]!=0 || isPeriodicFace(s,f)
        || (s.sstBoundaryKMode[f]!=2 && s.sstBoundaryOmegaMode[f]!=2))return;
    using Real=std::remove_reference_t<decltype(s.rho[0])>;
    Real mass=0,mx=0,my=0,mz=0,energy=0;
    if(computeRiemannGasFaceFluxDevice<false>(s,f,mass,mx,my,mz,energy,false,true))
        s.gasPhiRho[f]=mass;
}
