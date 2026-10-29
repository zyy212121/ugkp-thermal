#pragma once
#include "ParticleGeometryApi.H"
namespace {
__global__ void mapAttachedParticleGeometryKernel(DeviceState* sp,
    ugkp::thermal::ParticleSurfaceGeometry previous,ugkp::thermal::ParticleSurfaceGeometry endpoint,
    const int* faceMap) {
    DeviceState& s=*sp;
    if(!ugkp::diagnostics::blockMayProceed(s.diagnostics))return;
    const int count=*s.particleCountDevice;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x) {
        if(s.pStatus[i]!=1||s.pStuck[i]==Foam::gpuThermal::particleWallMobile)continue;
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
    s->preBaseDirectoryReady=0;s->splitPreDirectoryActive=0;
    return 0;
}
}}
