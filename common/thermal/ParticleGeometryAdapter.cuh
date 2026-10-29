#pragma once
#include "ParticleGeometryApi.H"
namespace {
// Relabel free particles after a moving internal face passes their unchanged
// physical position. External wall/inlet/outlet events stay with face transport.
__global__ void relocateMobileParticleCellsKernel(DeviceState* sp){
    DeviceState& s=*sp;
    const int count=*s.particleCountDevice;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x){
        if(!s.pStatus[i]||s.pStuck[i])continue;
        int cell=s.pCellId[i];
        for(int hop=0;hop<s.nCells&&cell>=0&&cell<s.nCells;++hop){
            if(pointInsideCell(s,cell,s.px[i],s.py[i],s.pz[i]))break;
            const int plane=mostViolatedPlane(s,cell,s.px[i],s.py[i],s.pz[i]);
            if(plane<0||s.cellFaceKind[plane]!=0)break;
            const int next=s.cellFaceNeighbor[plane];if(next<0||next==cell)break;
            cell=next;
        }
        s.pCellId[i]=cell;
    }
}
int refreshParticleGeometryMoments(DeviceState* s){
    relocateMobileParticleCellsKernel<<<s->particleWorkGrid,s->particleBlockThreads>>>(s->deviceState);
    const auto error=cudaGetLastError();if(error!=cudaSuccess){setLastError("relocate particles on refreshed mesh",error);return 1;}
    // Rebuild moments with new cell volumes but retain the accepted lagged gas
    // fraction, so the common volume source accounts for the geometry change.
    const int code=rebuildResidentParticleMomentsFromParticles(s,s->particleCapacity,false);
    if(!code)s->particleGeometryDirty=false;
    return code;
}
__global__ void refreshParticleTrackingGeometryKernel(DeviceState* sp,
    ugkp::thermal::ParticleTrackingBoundaryView boundary) {
    DeviceState& s=*sp;
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=s.nCells)return;
    const int begin=s.cellPlaneStart[c],end=begin+s.cellPlaneCount[c];
    for(int p=begin;p<end;++p) {
        const int f=s.cellFaceId[p];
        const auto plane=ugkp::thermal::particleTrackingPlane(c,s.faceOwner[f],
            s.Sfx[f],s.Sfy[f],s.Sfz[f],s.magSf[f],s.faceCx[f],s.faceCy[f],s.faceCz[f],GpuReal(OfSmall));
        s.planeNx[p]=plane.x;s.planeNy[p]=plane.y;s.planeNz[p]=plane.z;s.planeD[p]=plane.d;
        const bool internal=f<s.nInternalFaces;
        const int kind=internal?0:boundary.kind[f];
        s.cellFaceKind[p]=kind;
        s.cellFaceNeighbor[p]=internal?(s.faceOwner[f]==c?s.faceNeighbour[f]:s.faceOwner[f])
            :(kind==6?s.faceNeighbour[f]:-1);
        s.cellFaceRestitution[p]=internal?1.:boundary.restitution[f];
        s.cellFaceTangential[p]=internal?1.:boundary.tangential[f];
    }
}
__global__ void mapAttachedParticleGeometryKernel(DeviceState* sp,
    ugkp::thermal::ParticleSurfaceGeometry previous,ugkp::thermal::ParticleSurfaceGeometry endpoint,
    const int* faceMap) {
    DeviceState& s=*sp;
    if(!ugkp::diagnostics::blockMayProceed(s.diagnostics))return;
    const int count=*s.particleCountDevice;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x) {
        if(s.pStatus[i]==0||s.pStuck[i]==Foam::gpuThermal::particleWallMobile)continue;
        const int oldFace=s.pStuckFaceId[i];
        const int newFace=oldFace>=0&&oldFace<previous.nFaces?(faceMap?faceMap[oldFace]:oldFace):-1;
        const auto mapped=ugkp::thermal::mapAttachedParticleGeometry(previous,endpoint,oldFace,newFace,
            {s.px[i],s.py[i],s.pz[i]});
        if(!mapped.valid||mapped.cell>=s.nCells||mapped.face<s.nInternalFaces) {
            using namespace ugkp::diagnostics;
            auto record=makeRecord(Code::InvalidPhysicalState,30710,0,0,EntityKind::Particle,i,-1,oldFace,newFace);
            recordFatal(s.diagnostics,record);continue;
        }
        s.px[i]=mapped.position.x;s.py[i]=mapped.position.y;s.pz[i]=mapped.position.z;
        // Mesh remapping is not a force or a velocity update. In particular,
        // preserve the lab-frame incident velocity used by deferred rebound.
        s.pCellId[i]=mapped.cell;s.pStuckFaceId[i]=mapped.face;
    }
}
}
namespace ugkp { namespace thermal {
int refreshParticleTrackingGeometry(void* owner,const ParticleTrackingBoundaryView& boundary,cudaStream_t stream) {
    auto* s=asState(owner);
    if(validateState(s,"refresh resident particle tracking geometry"))return 1;
    if(boundary.nFaces!=s->nFaces || (s->nFaces>s->nInternalFaces
       && (!boundary.kind||!boundary.restitution||!boundary.tangential))) {
        setLastErrorText("particle tracking requires matching resident boundary topology/coefficients");return 1;
    }
    if(s->nCells) {
        refreshParticleTrackingGeometryKernel<<<(s->nCells+127)/128,128,0,stream>>>(s->deviceState,boundary);
        const auto error=cudaGetLastError();
        if(error!=cudaSuccess){setLastError("refresh resident particle tracking geometry",error);return 1;}
    }
    s->preBaseDirectoryReady=0;s->splitPreDirectoryActive=0;s->particleGeometryDirty=true;
    return 0;
}
int relocateMobileParticles(void* owner,cudaStream_t stream) {
    auto* s=asState(owner);
    if(validateState(s,"relocate mobile particles after mesh correction"))return 1;
    if(s->particleCapacity>0){
        relocateMobileParticleCellsKernel<<<s->particleWorkGrid,s->particleBlockThreads,0,stream>>>(s->deviceState);
        const auto error=cudaGetLastError();
        if(error!=cudaSuccess){setLastError("relocate mobile particles after mesh correction",error);return 1;}
    }
    s->preBaseDirectoryReady=0;s->splitPreDirectoryActive=0;s->particleGeometryDirty=true;
    return 0;
}
int mapAttachedParticles(void* owner,const ParticleSurfaceGeometry& previous,
    const ParticleSurfaceGeometry& endpoint,const int* faceMap,cudaStream_t stream) {
    auto* s=asState(owner);
    if(validateState(s,"map attached particle geometry"))return 1;
    if(endpoint.nFaces!=s->nFaces||previous.nFaces!=s->nFaces
       ||previous.nPoints<=0||endpoint.nPoints!=previous.nPoints
       ||!previous.faceOffsets||!previous.facePoints||!previous.points
       ||!endpoint.faceOffsets||!endpoint.facePoints||!endpoint.points||!endpoint.owner) {
        setLastErrorText("attached particle mapping requires matching, resident topology-preserving geometry");return 1;
    }
    if(s->particleCapacity>0) {
        mapAttachedParticleGeometryKernel<<<s->particleWorkGrid,s->particleBlockThreads,0,stream>>>
            (s->deviceState,previous,endpoint,faceMap);
        const auto error=cudaGetLastError();if(error!=cudaSuccess){setLastError("map attached particle geometry",error);return 1;}
    }
    s->preBaseDirectoryReady=0;s->splitPreDirectoryActive=0;s->particleGeometryDirty=true;
    return 0;
}
}}
