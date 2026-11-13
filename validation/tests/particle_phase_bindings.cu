// Binding/cache mechanism probe using the actual thermal backend functions.
// This does not replace the application physical/conservation comparisons.
#ifndef UGKP_PARTICLE_BACKEND_SOURCE
#define UGKP_PARTICLE_BACKEND_SOURCE "../../common/thermal/GpuResidentStrict.cu"
#endif
#include UGKP_PARTICLE_BACKEND_SOURCE
static void requireCuda(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}
static void requireBinding(int status){if(status){std::fprintf(stderr,"%s\n",lastError);std::exit(3);}}
__global__ void bindingObservation(DeviceState* s,GpuReal* out,int branch){
    if(!threadIdx.x&&!blockIdx.x)*out=s->gasCp+GpuReal(branch);
}
int main(){
    DeviceState s{};s.nCells=1;s.nFaces=1;s.nInternalFaces=0;s.nCellPlanes=0;
    s.borrowedGasFields=~std::uint64_t(0);s.borrowedGasSpecies=true;
    requireCuda(cudaMalloc(&s.deviceState,sizeof(s)));
    void* allocation=nullptr;void* replacement=nullptr;GpuReal* observed=nullptr;
    requireCuda(cudaMalloc(&allocation,1024));requireCuda(cudaMalloc(&replacement,1024));
    requireCuda(cudaMallocManaged(&observed,sizeof(GpuReal)));
    ugkp::thermal::ParticleGasBinding gas;gas.nCells=1;gas.nFaces=1;gas.nInternalFaces=0;
#define UGKP_PARTICLE_GAS_FIELD(Type,Name) gas.Name=static_cast<Type*>(allocation);
#include "ParticleGasFields.inc"
#undef UGKP_PARTICLE_GAS_FIELD
    gas.gasCp=GpuReal(1234);
    ugkp::thermal::ParticleSurfaceState surface{};
    surface.temperature=static_cast<const GpuReal*>(allocation);
    surface.velocityX=surface.velocityY=surface.velocityZ=surface.temperature;
    surface.effusivity=surface.temperature;
    ugkp::thermal::ParticleNumericalControls controls{};
    int captured=0;double end=0;
    auto advance=[&]{end+=.125;
        const int branch=int(s.gasSpecies.mode);
        auto submit=[&](double,double){++captured;
            bindingObservation<<<1,32,0,s.particleCaptureStream>>>(s.deviceState,observed,branch);
            return int(cudaGetLastError()!=cudaSuccess);};
        requireBinding(ugkp::gpu::launchParticlePhaseGraph(&s,.125,end,submit,[]{},
            [](const char* where,cudaError_t e){std::fprintf(stderr,"%s %s\n",where,cudaGetErrorString(e));}));
        requireCuda(cudaDeviceSynchronize());
        if(*observed!=gas.gasCp+GpuReal(branch))std::exit(4);
    };
    auto bind=[&]{requireBinding(ugkp::thermal::bindParticleGas(&s,gas));
        requireBinding(ugkp::thermal::bindParticleSurface(&s,1,surface));
        requireBinding(ugkp::thermal::setParticleNumericalControls(&s,controls));};
    for(int i=0;i<8;++i){bind();advance();}
    const int stableCaptures=captured;
    // Real borrowed-address, host model branch, surface and control changes.
    gas.rho=static_cast<GpuReal*>(replacement);bind();advance();
    gas.species.mode=ugkwp::GasMode::MixtureFrozen;bind();advance();
    surface.temperature=static_cast<const GpuReal*>(replacement);bind();advance();
    ++controls.packingIterations;bind();advance();
    std::printf("BIND precision=%zu stableCaptures=%d totalCaptures=%d epoch=%llu\n",
        sizeof(GpuReal)*8,stableCaptures,captured,s.particleGraphEpoch);
#ifdef UGKP_EXPECT_REUSE
    if(stableCaptures!=1||captured!=5)return 5;
#else
    // Previous owner intentionally invalidates on each of the three binds.
    if(stableCaptures!=8||captured!=12)return 6;
#endif
    ugkp::gpu::releaseParticlePhaseGraphs(s.particlePhaseGraphs);
    if(s.particleGasReady)requireCuda(cudaEventDestroy(s.particleGasReady));
    if(s.particleGasDone)requireCuda(cudaEventDestroy(s.particleGasDone));
    requireCuda(cudaFree(s.deviceState));requireCuda(cudaFree(allocation));
    requireCuda(cudaFree(replacement));requireCuda(cudaFree(observed));
    return 0;
}
