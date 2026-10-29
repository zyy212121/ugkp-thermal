#pragma once
#include "ParticleStateApi.H"
namespace {
using ParticleSnapshot=ugkp::thermal::ParticleState<>;
__global__ void uploadParticleSnapshotKernel(DeviceState* sp,int begin,int count,const ParticleSnapshot* in) {
    DeviceState& s=*sp;
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<count;k+=blockDim.x*gridDim.x) {
        const int i=begin+k;const ParticleSnapshot& q=in[k];
#define UGKP_PARTICLE_FIELD(Field,Value) s.Field[i]=q.Value;
#define UGKP_PARTICLE_OPTIONAL(Field,Value) if(s.Field)s.Field[i]=q.Value;
#define UGKP_PARTICLE_ARRAY(Field,Value,N) if(s.Field)for(int j=0;j<N;++j)s.Field[i*N+j]=q.Value[j];
#include "ParticleStateFields.inc"
#undef UGKP_PARTICLE_FIELD
#undef UGKP_PARTICLE_OPTIONAL
#undef UGKP_PARTICLE_ARRAY
    }
}
__global__ void downloadParticleSnapshotKernel(DeviceState* sp,int begin,int count,ParticleSnapshot* out) {
    const DeviceState& s=*sp;
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<count;k+=blockDim.x*gridDim.x) {
        const int i=begin+k;ParticleSnapshot q{};
#define UGKP_PARTICLE_FIELD(Field,Value) q.Value=s.Field[i];
#define UGKP_PARTICLE_OPTIONAL(Field,Value) if(s.Field)q.Value=s.Field[i];
#define UGKP_PARTICLE_ARRAY(Field,Value,N) if(s.Field)for(int j=0;j<N;++j)q.Value[j]=s.Field[i*N+j];
#include "ParticleStateFields.inc"
#undef UGKP_PARTICLE_FIELD
#undef UGKP_PARTICLE_OPTIONAL
#undef UGKP_PARTICLE_ARRAY
        q.thermal.initialized=true;
        q.energy=q.mass*particleEffectiveSpecificEnthalpyDevice(s,i);
        // Saved velocities are lab-frame delayed rebound velocities in the
        // existing contact model. These two fields are derived diagnostics.
        if(q.state && q.face>=0 && q.face<s.nFaces) {
            q.contactRelativeVelocity={q.thermal.savedVelocity[0]-ugkwp::particleWallVelocityX(s,q.face),
                q.thermal.savedVelocity[1]-ugkwp::particleWallVelocityY(s,q.face),
                q.thermal.savedVelocity[2]-ugkwp::particleWallVelocityZ(s,q.face)};

        }
        ugkp::thermal::normalizeParticleContactMechanicalEnergy(q);
        out[k]=q;
    }
}
struct ParticleSnapshotBuffer {
    ParticleSnapshot* device=nullptr;
    ~ParticleSnapshotBuffer(){if(device)cudaFree(device);}
};
constexpr int particleSnapshotChunk=65536;
}
namespace ugkp { namespace thermal {
int uploadParticleState(void* owner,int count,const ParticleState<>* particles) {
    DeviceState* s=asState(owner);
    if(validateState(s,"complete particle state upload"))return 1;
    if(count<0 || count>s->particleCapacity || (count && !particles)) {
        setLastErrorText("invalid complete particle state upload count/storage");return 1;
    }
    for(int i=0;i<count;++i) {
        const auto& p=particles[i];
        if(!p.thermal.initialized || p.state<0 || p.state>3 ||
           (p.thermal.status && (p.cell<0 || p.cell>=s->nCells)) ||
           (p.state && (p.face<0 || p.face>=s->nFaces))) {
            setLastErrorText("complete particle state requires initialized history and valid mesh association");return 1;
        }
        if(p.state && !s->particleStuckModelConfigured) {
            setLastErrorText("attached particle state requires configured shared contact model");return 1;
        }
#define UGKP_PARTICLE_FIELD(Field,Value)
#define UGKP_PARTICLE_OPTIONAL(Field,Value) if(!s->Field && p.Value!=0) {setLastErrorText("missing resident storage for " #Field);return 1;}
#define UGKP_PARTICLE_ARRAY(Field,Value,N) if(!s->Field)for(int j=0;j<N;++j)if(p.Value[j]!=0) {setLastErrorText("missing resident storage for " #Field);return 1;}
#include "ParticleStateFields.inc"
#undef UGKP_PARTICLE_FIELD
#undef UGKP_PARTICLE_OPTIONAL
#undef UGKP_PARTICLE_ARRAY
    }
    ParticleSnapshotBuffer buffer;
    if(count && allocate(buffer.device,std::min(count,particleSnapshotChunk),"particle state staging"))return 1;
    if(syncDeviceState(s,"particle state upload view"))return 1;
    cudaError_t err=cudaMemset(s->pStatus,0,size_t(s->particleCapacity)*sizeof(*s->pStatus));
    if(err!=cudaSuccess){setLastError("particle state status reset",err);return 1;}
    for(int begin=0;begin<count;) {
        const int n=std::min(count-begin,particleSnapshotChunk);
        if(copyToDevice(buffer.device,particles+begin,n,"particle state upload chunk"))return 1;
        uploadParticleSnapshotKernel<<<(n+255)/256,256>>>(s->deviceState,begin,n,buffer.device);
        err=cudaGetLastError();if(err!=cudaSuccess){setLastError("particle state unpack",err);return 1;}
        begin+=n;
    }
    if(copyToDevice(s->particleCountDevice,&count,1,"particle state count"))return 1;
    s->preBaseDirectoryReady=0;s->splitPreDirectoryActive=0;s->particlesMayBePresent=count!=0 || s->nBoundarySources>0;
    err=cudaMemset(s->wallBoundParticleCountDevice,0,sizeof(int));
    if(err!=cudaSuccess){setLastError("particle wall directory reset",err);return 1;}
    if(count)rebuildWallBoundParticleDirectoryKernel<<<s->particleWorkGrid,s->particleBlockThreads>>>(s->deviceState);
    err=cudaGetLastError();if(err!=cudaSuccess){setLastError("particle wall directory rebuild",err);return 1;}
    const int code=rebuildResidentParticleMomentsFromParticles(s,count);
    if(!code)s->particleGeometryDirty=false;
    return code;
}
int downloadParticleState(void* owner,std::vector<ParticleState<>>& particles) {
    DeviceState* s=asState(owner);
    if(validateState(s,"complete particle state download"))return 1;
    int count=0;if(copyToHost(&count,s->particleCountDevice,1,"particle state download count"))return 1;
    if(count<0 || count>s->particleCapacity){setLastErrorText("invalid resident particle count");return 1;}
    std::vector<ParticleState<>> candidate(count);
    ParticleSnapshotBuffer buffer;
    if(count && allocate(buffer.device,std::min(count,particleSnapshotChunk),"particle state staging"))return 1;
    for(int begin=0;begin<count;) {
        const int n=std::min(count-begin,particleSnapshotChunk);
        downloadParticleSnapshotKernel<<<(n+255)/256,256>>>(s->deviceState,begin,n,buffer.device);
        const auto err=cudaGetLastError();if(err!=cudaSuccess){setLastError("particle state pack",err);return 1;}
        if(copyToHost(candidate.data()+begin,buffer.device,n,"particle state download chunk"))return 1;
        begin+=n;
    }
    const int rc=s->diagnosticOwner->completeReadback(0,count!=0);
    if(rc==0)particles.swap(candidate);
    return rc;
}
}}
