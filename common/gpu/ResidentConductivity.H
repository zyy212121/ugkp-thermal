#pragma once
#include "../GpuPrecisionTypes.H"
#include "ThermalFieldOperators.H"

namespace ugkp { namespace gpu {
enum class ConductivityBoundary { Supplied, Coupled, Owner };

#ifdef __CUDACC__
namespace diffusionDetail {
static __global__ void interpolateResidentConductivity(int faces,int internalFaces,
    const int* owner,const int* neighbour,const GpuReal* cells,const GpuReal* weights,
    const GpuReal* boundary,const GpuReal* adjacent,const int* boundarySource,
    const int* adjacentCells,const GpuReal* jump,const GpuReal* boundaryDevice,bool harmonic,GpuReal* gamma){
    const int f=blockIdx.x*blockDim.x+threadIdx.x;
    if(f>=faces)return;
    const GpuReal own=cells[owner[f]];
    if(f<internalFaces){
        gamma[f]=thermalScalarInterpolation({own,cells[neighbour[f]],weights[f],0},harmonic);
        return;
    }
    const int b=f-internalFaces;
    const auto source=static_cast<ConductivityBoundary>(boundarySource[b]);
    if(source==ConductivityBoundary::Coupled){
        const GpuReal other=adjacentCells[b]>=0?cells[adjacentCells[b]]-jump[b]:adjacent[b];
        gamma[f]=thermalScalarInterpolation({own,other,weights[f],1},harmonic);
    }else{
        gamma[f]=source==ConductivityBoundary::Owner?own:boundaryDevice?boundaryDevice[b]:boundary[b];
    }
}
}
#endif
}}
