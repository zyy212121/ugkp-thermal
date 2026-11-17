#ifndef UGKP_GPU_PHYSICAL_DIFFUSION_OPERATOR_CUH
#define UGKP_GPU_PHYSICAL_DIFFUSION_OPERATOR_CUH
#include "SparseConjugateGradient.H"
namespace ugkp { namespace gpu { namespace physicalDiffusionDetail {
__device__ inline float internal(const SparsePhysicalDiffusionValues& v,int face){
    const float g=v.gamma[face]*v.area[face];
    const float coefficient=v.delta[face]*g;
    return v.transient?-coefficient:coefficient;
}
__device__ inline float boundaryDiagonal(const SparsePhysicalDiffusionValues& v,int face){
    const float g=v.gamma[v.internalFaces+face]*v.area[v.internalFaces+face];
    const float coefficient=g*v.gradientInternal[face];
    return v.transient?float(0)-coefficient:coefficient;
}
__device__ inline float boundarySource(const SparsePhysicalDiffusionValues& v,int face){
    const float g=v.gamma[v.internalFaces+face]*v.area[v.internalFaces+face];
    const float coefficient=(-g)*v.gradientBoundary[face];
    return v.transient?float(0)-coefficient:coefficient;
}
__device__ inline float diagonal(const SparsePhysicalDiffusionValues& v,int row){
    float value=float(0);
    for(int e=v.cellOffsets[row];e<v.cellOffsets[row+1];++e){
        const int f=v.cellFaces[e];if(f<v.internalFaces){const float g=v.gamma[f]*v.area[f];value-=v.delta[f]*g;}
    }
    if(v.transient)value=v.volume[row]*(v.capacity[row]/(*v.interval))-value;
    return value;
}
__device__ inline float completedDiagonal(const SparsePhysicalDiffusionValues& v,int row){
    float value=diagonal(v,row);
    for(int e=v.cellOffsets[row];e<v.cellOffsets[row+1];++e){const int f=v.cellFaces[e];if(f>=v.internalFaces)value+=boundaryDiagonal(v,f-v.internalFaces);}
    return value;
}
}}}
#endif
