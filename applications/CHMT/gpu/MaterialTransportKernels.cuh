#ifndef CHMT_GPU_MATERIALTRANSPORTKERNELS_CUH
#define CHMT_GPU_MATERIALTRANSPORTKERNELS_CUH
#include "materials/MaterialTransportMath.H"
#include <cuda_runtime.h>
namespace chmt { namespace materialTransportGpu {
static __global__ void initialize(const PhysicsConfig* p,Real interval,Real cfl,int cells,
    MaterialTransportDeviceResult* result) {
    *result=MaterialTransportDeviceResult{};
    result->numerical=true; // Same initial failure classification as the host API.
    if(!(interval>0)||!finite(interval)||!(finite(cfl)&&cfl>0)){
        result->errorCode=int(MaterialTransportError::InvalidControl);result->numerical=true;return;}
    if(cells==0)return;
    for(int species=0;species<Ns;++species)
        if(!finite(p->poreEffectiveDiffusivity[species])||p->poreEffectiveDiffusivity[species]<0){
            result->errorCode=int(MaterialTransportError::InvalidDiffusivity);
            result->entity=species;result->numerical=true;return;}
}
static __global__ void recover(MaterialMeshView mesh,const SolidQ* q,const PhysicsConfig* p,
    MaterialTransportWorkspace w,const MaterialTransportDeviceResult* result) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=mesh.cells||result->errorCode)return;
    w.cellErrors[c]=int(materialTransport::recoverCell(mesh,q,*p,c,w));
}
static __global__ void pressure(MaterialMeshView mesh,const PhysicsConfig* p,
    MaterialTransportWorkspace w,const MaterialTransportDeviceResult* result) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=mesh.cells||result->errorCode)return;
    w.cellErrors[c]=int(materialTransport::pressureCell(mesh,*p,c,w));
}
static __global__ void stability(MaterialMeshView mesh,const SolidQ* q,const PhysicsConfig* p,
    const Real* swept,Real interval,MaterialTransportWorkspace w,const MaterialTransportDeviceResult* result) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=mesh.cells||result->errorCode)return;
    w.cellErrors[c]=int(materialTransport::stabilityCell(mesh,q,*p,swept,interval,c,w));
}
static __global__ void faces(MaterialMeshView mesh,const SolidQ* q,const PhysicsConfig* p,
    const Real* swept,Real interval,MaterialTransportWorkspace w,const MaterialTransportDeviceResult* result) {
    const int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=mesh.faces||result->errorCode)return;
    w.faceErrors[f]=int(materialTransport::face(mesh,q,*p,swept,interval,f,w));
}
// A deterministic compact record: minimum failing entity in this completed
// stage. No racing first-error writer and no host per-cell diagnostics copy.
static __global__ void compactErrors(const int* errors,int count,MaterialTransportDeviceResult* result) {
    if(result->errorCode)return;
    __shared__ int first[128];
    int entity=count;
    for(int i=threadIdx.x;i<count;i+=blockDim.x)if(errors[i]){entity=i;break;}
    first[threadIdx.x]=entity;__syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride/=2){
        if(threadIdx.x<stride&&first[threadIdx.x+stride]<first[threadIdx.x])first[threadIdx.x]=first[threadIdx.x+stride];
        __syncthreads();
    }
    if(threadIdx.x==0&&first[0]<count){result->entity=first[0];result->errorCode=errors[first[0]];result->numerical=true;}
}
static __global__ void gather(MaterialMeshView mesh,const SolidQ* q,const PhysicsConfig* p,
    Real cfl,MaterialTransportWorkspace w,const MaterialTransportDeviceResult* result) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=mesh.cells||result->errorCode)return;
    materialTransport::gatherCell(mesh,q,*p,cfl,c,w);
}
static __global__ void compactTimestep(const Real* dt,int count,MaterialTransportDeviceResult* result) {
    if(result->errorCode)return;
    __shared__ Real limits[128];
    Real limit=std::numeric_limits<Real>::max();
    for(int i=threadIdx.x;i<count;i+=blockDim.x)limit=minValue(limit,dt[i]);
    limits[threadIdx.x]=limit;__syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride/=2){
        if(threadIdx.x<stride)limits[threadIdx.x]=minValue(limits[threadIdx.x],limits[threadIdx.x+stride]);
        __syncthreads();
    }
    if(threadIdx.x==0){
        result->dtLimit=limits[0];
        result->numerical=false;
        if(!finite(limits[0])||limits[0]<=0){result->errorCode=int(MaterialTransportError::InvalidTimestep);result->numerical=true;}
    }
}
} // namespace materialTransportGpu
// All pointers, including p and result, refer to device storage. Geometry and
// workspaces remain owned by the caller; stream completion bounds their use.
inline cudaError_t launchMaterialTransport(MaterialMeshView mesh,const SolidQ* q,
    const PhysicsConfig* p,const Real* swept,Real interval,Real cfl,
    MaterialTransportWorkspace workspace,MaterialTransportDeviceResult* result,cudaStream_t stream) {
    using namespace materialTransportGpu;
    initialize<<<1,1,0,stream>>>(p,interval,cfl,mesh.cells,result);
    if(mesh.cells==0)return cudaGetLastError();
    constexpr int threads=128;
    const int cells=(mesh.cells+threads-1)/threads,faceBlocks=(mesh.faces+threads-1)/threads;
    recover<<<cells,threads,0,stream>>>(mesh,q,p,workspace,result);
    compactErrors<<<1,threads,0,stream>>>(workspace.cellErrors,mesh.cells,result);
    pressure<<<cells,threads,0,stream>>>(mesh,p,workspace,result);
    compactErrors<<<1,threads,0,stream>>>(workspace.cellErrors,mesh.cells,result);
    stability<<<cells,threads,0,stream>>>(mesh,q,p,swept,interval,workspace,result);
    compactErrors<<<1,threads,0,stream>>>(workspace.cellErrors,mesh.cells,result);
    if(mesh.faces){
        faces<<<faceBlocks,threads,0,stream>>>(mesh,q,p,swept,interval,workspace,result);
        compactErrors<<<1,threads,0,stream>>>(workspace.faceErrors,mesh.faces,result);
    }
    gather<<<cells,threads,0,stream>>>(mesh,q,p,cfl,workspace,result);
    compactTimestep<<<1,threads,0,stream>>>(workspace.cellDtLimit,mesh.cells,result);
    return cudaGetLastError();
}
} // namespace chmt
#endif
