#ifndef UGKP_GPU_PHYSICAL_DIFFUSION_OPERATOR_CUH
#define UGKP_GPU_PHYSICAL_DIFFUSION_OPERATOR_CUH
#include "SparseConjugateGradient.H"
namespace ugkp { namespace gpu { namespace physicalDiffusionDetail {
__device__ inline double internal(const SparsePhysicalDiffusionValues& v,int face){
    const double g=v.gamma[face]*v.area[face];
    const double coefficient=v.delta[face]*g;
    return v.transient?-coefficient:coefficient;
}
__device__ inline double boundaryDiagonal(const SparsePhysicalDiffusionValues& v,int face){
    const double g=v.gamma[v.internalFaces+face]*v.area[v.internalFaces+face];
    const double coefficient=g*v.gradientInternal[face];
    return v.transient?double(0)-coefficient:coefficient;
}
__device__ inline double boundarySource(const SparsePhysicalDiffusionValues& v,int face){
    const double g=v.gamma[v.internalFaces+face]*v.area[v.internalFaces+face];
    const double coefficient=(-g)*v.gradientBoundary[face];
    return v.transient?double(0)-coefficient:coefficient;
}
__device__ inline double diagonal(const SparsePhysicalDiffusionValues& v,int row){
    double value=double(0);
    for(int e=v.cellOffsets[row];e<v.cellOffsets[row+1];++e){
        const int f=v.cellFaces[e];if(f<v.internalFaces){const double g=v.gamma[f]*v.area[f];value-=v.delta[f]*g;}
    }
    if(v.transient)value=v.volume[row]*(v.capacity[row]/(*v.interval))-value;
    return value;
}
__device__ inline double completedDiagonal(const SparsePhysicalDiffusionValues& v,int row){
    double value=diagonal(v,row);
    for(int e=v.cellOffsets[row];e<v.cellOffsets[row+1];++e){const int f=v.cellFaces[e];if(f>=v.internalFaces)value+=boundaryDiagonal(v,f-v.internalFaces);}
    return value;
}
}}}
#endif
