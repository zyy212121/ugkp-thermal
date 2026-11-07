#ifndef CHMT_GPU_MATERIALTRANSPORTKERNELS_CUH
#define CHMT_GPU_MATERIALTRANSPORTKERNELS_CUH
#include "materials/MaterialTransportMath.H"
#include <cuda_runtime.h>
#include "../../../common/gpu/DeviceValue.cuh"
namespace chmt { namespace materialTransportGpu {
template<class Interval,class Cfl>
static __global__ void initialize(const PhysicsConfig* p,Interval intervalValue,Cfl cflValue,int cells,
    MaterialTransportDeviceResult* result) {
    const Real interval=intervalValue,cfl=cflValue;
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
template<class Mesh,class State>
static __global__ void recover(Mesh meshValue,State q,const PhysicsConfig* p,
    MaterialTransportWorkspace w,const MaterialTransportDeviceResult* result) {
    const MaterialMeshView mesh=meshValue;
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=mesh.cells||result->errorCode)return;
    w.cellErrors[c]=int(materialTransport::recoverCell(mesh,q,*p,c,w));
}
template<class Mesh>
static __global__ void pressure(Mesh meshValue,const PhysicsConfig* p,
    MaterialTransportWorkspace w,const MaterialTransportDeviceResult* result) {
    const MaterialMeshView mesh=meshValue;
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=mesh.cells||result->errorCode)return;
    w.cellErrors[c]=int(materialTransport::pressureCell(mesh,*p,c,w));
}
template<class Mesh,class State,class Sweep,class Interval>
static __global__ void stability(Mesh meshValue,State q,const PhysicsConfig* p,
    Sweep swept,Interval interval,MaterialTransportWorkspace w,const MaterialTransportDeviceResult* result) {
    const MaterialMeshView mesh=meshValue;
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=mesh.cells||result->errorCode)return;
    w.cellErrors[c]=int(materialTransport::stabilityCell(mesh,q,*p,swept,interval,c,w));
}
template<class Mesh,class State,class Sweep,class Interval>
static __global__ void faces(Mesh meshValue,State q,const PhysicsConfig* p,
    Sweep swept,Interval interval,MaterialTransportWorkspace w,const MaterialTransportDeviceResult* result) {
    const MaterialMeshView mesh=meshValue;
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
template<class Mesh,class State,class Cfl>
static __global__ void gather(Mesh meshValue,State q,const PhysicsConfig* p,
    Cfl cfl,MaterialTransportWorkspace w,const MaterialTransportDeviceResult* result) {
    const MaterialMeshView mesh=meshValue;
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
template<class Mesh,class State,class Sweep,class Interval,class Cfl>
inline cudaError_t launchMaterialTransport(int cellCount,int faceCount,Mesh mesh,State q,
    const PhysicsConfig* p,Sweep swept,Interval interval,Cfl cfl,
    MaterialTransportWorkspace workspace,MaterialTransportDeviceResult* result,cudaStream_t stream) {
    using namespace materialTransportGpu;
    initialize<<<1,1,0,stream>>>(p,interval,cfl,cellCount,result);
    if(cellCount==0)return cudaGetLastError();
    constexpr int threads=128;
    const int cells=(cellCount+threads-1)/threads,faceBlocks=(faceCount+threads-1)/threads;
    recover<<<cells,threads,0,stream>>>(mesh,q,p,workspace,result);
    compactErrors<<<1,threads,0,stream>>>(workspace.cellErrors,cellCount,result);
    pressure<<<cells,threads,0,stream>>>(mesh,p,workspace,result);
    compactErrors<<<1,threads,0,stream>>>(workspace.cellErrors,cellCount,result);
    stability<<<cells,threads,0,stream>>>(mesh,q,p,swept,interval,workspace,result);
    compactErrors<<<1,threads,0,stream>>>(workspace.cellErrors,cellCount,result);
    if(faceCount){
        faces<<<faceBlocks,threads,0,stream>>>(mesh,q,p,swept,interval,workspace,result);
        compactErrors<<<1,threads,0,stream>>>(workspace.faceErrors,faceCount,result);
    }
    gather<<<cells,threads,0,stream>>>(mesh,q,p,cfl,workspace,result);
    compactTimestep<<<1,threads,0,stream>>>(workspace.cellDtLimit,cellCount,result);
    return cudaGetLastError();
}
// Existing immutable-value callers share the same launch sequence. Resident
// owners pass topology counts plus DeviceValue bindings for geometry/state;
// no accepted pointer or numerical mesh value has to be returned to the host.
template<class State,class Sweep,class Interval,class Cfl>
inline cudaError_t launchMaterialTransport(MaterialMeshView mesh,State q,
    const PhysicsConfig* p,Sweep swept,Interval interval,Cfl cfl,
    MaterialTransportWorkspace workspace,MaterialTransportDeviceResult* result,cudaStream_t stream){
    return launchMaterialTransport(mesh.cells,mesh.faces,mesh,q,p,swept,interval,cfl,workspace,result,stream);
}
} // namespace chmt
#endif
