// Shared seed and actual injection-kernel regression. The GPU fixture supplies
// one source and a planar unit face; it does not replace full solver validation.
#include "BoundaryInjectionSequence.H"
#include "BoundaryInjectionApi.H"
#include "../../applications/gasUGKPSteady/SteadyRestartState.H"
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>
struct SeedState { ugkwp::ParticleResidentInterval particleResidentInterval{}; };
static void cpuChecks(const std::string& path) {
    SeedState state;
    std::set<unsigned long long> keys;
    for(unsigned long long i=1;i<=64;++i) {
        const auto key=ugkwp::particleInjectionPositionSeed(state,123,0,.5,0,i,true);
        keys.insert(key);
        state.particleResidentInterval={.5,.5,true};
        state.particleResidentInterval.positionSequenceEnabled=true;
        state.particleResidentInterval.positionSequence=i;
        assert(key==ugkwp::particleInjectionPositionSeed(state,123,0,.5,0,999,true));
        state.particleResidentInterval.active=false;
        assert(ugkwp::particleInjectionPositionSeed(state,123,0,.5,0)==
            ugkwp::injectionPositionSeed(123,0,.5,0));
    }
    assert(keys.size()==64);
    // Face identity and sample identity must not share an interchangeable hash
    // domain: exchanging face=1/sequence=2 with face=2/sequence=1 is distinct.
    assert(ugkwp::particleInjectionPositionSeed(state,123,1,.5,0,2,true)!=
        ugkwp::particleInjectionPositionSeed(state,123,2,.5,0,1,true));
    state.particleResidentInterval.active=true;
    state.particleResidentInterval.positionSequenceEnabled=false;
    assert(ugkwp::particleInjectionPositionSeed(state,123,1,.5,0,2,true)==
        ugkwp::injectionPositionSeed(123,1,.5,0));
    state.particleResidentInterval.active=false;
    struct LegacyState {};
    assert(ugkwp::particleInjectionPositionSeed(LegacyState{},123,1,.5,0)==
        ugkwp::injectionPositionSeed(123,1,.5,0));
    // Resume the next sequence from unchanged steadyState.dat wire fields.
    SteadyRestartState saved{.5,16.,32,.5},loaded;
    assert(writeSteadyRestartState(path,saved));
    assert(readSteadyRestartState(path,loaded));
    for(unsigned long long i=1;i<=32;++i)
        assert(ugkwp::particleInjectionPositionSeed(state,123,0,.5,0,loaded.sampleCount+i,true)==
            ugkwp::particleInjectionPositionSeed(state,123,0,.5,0,32+i,true));
    saved.sampleCount=(1ULL<<53)+17;
    assert(writeSteadyRestartState(path,saved));
    assert(readSteadyRestartState(path,loaded)&&loaded.sampleCount==saved.sampleCount);
    assert(ugkwp::particleInjectionPositionSeed(state,123,0,.5,0,saved.sampleCount,true)!=
        ugkwp::particleInjectionPositionSeed(state,123,0,.5,0,saved.sampleCount+1,true));
    SteadyRestartState missing;
    assert(!readSteadyRestartState(path+".missing",missing)&&missing.sampleCount==0);
    std::remove(path.c_str());
    std::puts("PASS CPU 64 independent sequence keys, face/sequence domain separation, resident precedence, legacy time seed, 32+32 restart, uint64 above 2^53, missing metadata fallback");
}
#ifdef __CUDACC__
#include <cuda_runtime.h>
static int parameterUpdates=0;
static cudaError_t countedSetParams(cudaGraphExec_t graph,cudaGraphNode_t node,const cudaKernelNodeParams* p) {
    ++parameterUpdates;return cudaGraphExecKernelNodeSetParams(graph,node,p);
}
#define cudaGraphExecKernelNodeSetParams countedSetParams
#include "GpuParticlePhaseGraph.cuh"
#undef cudaGraphExecKernelNodeSetParams
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_R(x) x
#define GPU_OPERATOR_THERMAL 0
constexpr int capacity=256;
struct DeviceState {
    DeviceState* deviceState=nullptr;
    cudaStream_t particleCaptureStream=nullptr;cudaGraph_t particleCaptureRoot=nullptr;
    ugkwp::ParticleResidentInterval particleResidentInterval{};
    ugkp::gpu::ParticlePhaseGraphs particlePhaseGraphs{};
    unsigned long long particleGraphEpoch=1;
    int preBaseDirectoryReady=0,splitPreDirectoryActive=0,csrTasksReady=0,csrPreparedDirectoryKind=0;
    int nBoundarySources=1,nFaces=1,nCells=1,particleCapacity=capacity,count=0;
    int* particleCountDevice=nullptr;
    int sourceFace[1]{0},sourceCell[1]{0},offsets[2]{0,2},triangleCounts[1]{2};
    bool particleInjectionVelocitySpecified=false;
    double particleInjectionVelocity[3]{},injectionParcelMass=1,rhoSolid=1,TpMin=1,TpMax=1000,particleDiameterFallback=1;
    double gasBoundaryUx[1]{1},gasBoundaryUy[1]{},gasBoundaryUz[1]{},theta[1]{};
    double sourceUx[1]{1},sourceUy[1]{},sourceUz[1]{},sourceTheta[1]{},sourceT[1]{300},sourceD[1]{1};
    double Sfx[1]{-1},Sfy[1]{},Sfz[1]{},sourceMassRate[1]{2},sourceResidualMass[1]{};
    double sourcePx[1]{},sourcePy[1]{.5},sourcePz[1]{.5},Cx[1]{.5},Cy[1]{.5},Cz[1]{.5};
    double px[capacity]{},py[capacity]{},pz[capacity]{},pux[capacity]{},puy[capacity]{},puz[capacity]{},pT[capacity]{},pTheta[capacity]{},pd[capacity]{},pm[capacity]{};
    int pCellId[capacity]{},pStatus[capacity]{};
    unsigned long long rngSeed=123,pRng[capacity]{},pOrigId[capacity]{};
    ugkwp::BoundaryInjectionStorage boundaryInjection;
    ugkwp::BoundaryInjectionTriangle triangles[2]{
        {{0,0,0},{0,1,0},{0,0,1},.5},
        {{0,1,1},{0,-1,0},{0,0,-1},1}};
};
__device__ double finiteOr(double value,double fallback){return isfinite(value)?value:fallback;}
__device__ double clampMin(double value,double lo){return value<lo?lo:value;}
__device__ double clampRange(double value,double lo,double hi){return value<lo?lo:value>hi?hi:value;}
__device__ bool scheduledInletFaceDevice(const DeviceState&,int){return true;}
__device__ double scheduledSolidVolumeFractionDevice(const DeviceState&,double time){return 1+2*time;}
__device__ unsigned long long mixSeed(unsigned long long value){return ugkwp::injectionMix(value);}
__device__ double sampleDiameterAroundDevice(const DeviceState&,double diameter,unsigned long long& rng){rng=mixSeed(rng);return diameter+double(rng>>12)*1e-19;}
__device__ void resetSeparateContactAge(DeviceState&,int,double){}
#include "operators/injectBoundaryParticlesKernel.cuh"
static void check(cudaError_t error){if(error!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::abort();}}
struct Result {std::vector<double> y,z,diameter,mass;std::vector<unsigned long long> rng;int recordings=0,updates=0;};
static Result run(int mode,unsigned long long completed,int samples,bool republish=false) {
    DeviceState state;
    check(cudaMalloc(&state.deviceState,sizeof(state)));
    auto* base=reinterpret_cast<unsigned char*>(state.deviceState);
    state.particleCountDevice=reinterpret_cast<int*>(base+offsetof(DeviceState,count));
    state.boundaryInjection.offsets=reinterpret_cast<int*>(base+offsetof(DeviceState,offsets));
    state.boundaryInjection.counts=reinterpret_cast<int*>(base+offsetof(DeviceState,triangleCounts));
    state.boundaryInjection.triangles=reinterpret_cast<ugkwp::BoundaryInjectionTriangle*>(base+offsetof(DeviceState,triangles));
    state.particlePhaseGraphs.positionSequenceEnabled=mode!=0;
    state.particlePhaseGraphs.positionSequence=completed;
    check(cudaMemcpy(state.deviceState,&state,sizeof(state),cudaMemcpyHostToDevice));
    Result result;const int updateStart=parameterUpdates;
    for(int step=0;step<samples;++step) {
        if(republish&&step==samples/2) {
            // Preserve authoritative particle arrays while intentionally sending
            // a stale host interval, as full configuration publication does.
            DeviceState actual;check(cudaMemcpy(&actual,state.deviceState,sizeof(actual),cudaMemcpyDeviceToHost));
            actual.particleResidentInterval=state.particleResidentInterval;
            actual.particlePhaseGraphs=state.particlePhaseGraphs;
            actual.particleGraphEpoch=++state.particleGraphEpoch;
            check(cudaMemcpy(state.deviceState,&actual,sizeof(actual),cudaMemcpyHostToDevice));
        }
        if(mode<2) {
            if(mode==0)injectBoundaryParticlesKernel<<<1,1>>>(state.deviceState,.5,.5);
            else injectBoundaryParticlesKernel<<<1,1>>>(state.deviceState,.5,.5,completed+step+1,true);
            check(cudaGetLastError());
        } else {
            const int status=ugkp::gpu::launchParticlePhaseGraph(&state,.5,.5,
                [&](double dt,double time){
                    ++result.recordings;
                    // An intentionally different explicit key proves that the
                    // resident interval owns graph replay, not captured inputs.
                    injectBoundaryParticlesKernel<<<1,1,0,state.particleCaptureStream>>>(state.deviceState,dt,time,999,true);
                    state.preBaseDirectoryReady=1;
                    return int(cudaGetLastError()!=cudaSuccess);
                },[]{},[&](const char* label,cudaError_t e){std::fprintf(stderr,"%s: %s\n",label,cudaGetErrorString(e));});
            assert(status==0);
            const auto& entry=state.particlePhaseGraphs.entry[step==0?0:1];
            if(step>0)assert(entry.sequenceAdvances&&entry.parameters.positionSequence==0&&!entry.advances);
        }
    }
    DeviceState actual;check(cudaMemcpy(&actual,state.deviceState,sizeof(actual),cudaMemcpyDeviceToHost));
    assert(actual.count==samples&&actual.sourceResidualMass[0]==0);
    for(int i=0;i<samples;++i){
        assert(actual.pm[i]==1&&actual.pux[i]==1&&actual.puy[i]==0&&actual.puz[i]==0);
        assert(actual.px[i]==0&&actual.py[i]>=0&&actual.py[i]<=1&&actual.pz[i]>=0&&actual.pz[i]<=1);
        result.y.push_back(actual.py[i]);result.z.push_back(actual.pz[i]);
        result.diameter.push_back(actual.pd[i]);result.mass.push_back(actual.pm[i]);result.rng.push_back(actual.pRng[i]);
    }
    if(mode>=2){
        assert(actual.particleResidentInterval.end==.5&&!actual.particleResidentInterval.active);
        assert(actual.particleResidentInterval.positionSequence==completed+samples);
        assert(state.particlePhaseGraphs.positionSequence==completed+samples);
        result.updates=parameterUpdates-updateStart;
        // One warm-up entry and one steady entry, with no per-step updates.
        assert(result.recordings==(republish?3:2)&&result.updates==0);
    }
    ugkp::gpu::releaseParticlePhaseGraphs(state.particlePhaseGraphs);
    check(cudaFree(state.deviceState));return result;
}
static void gpuChecks(){
    const auto legacy=run(0,0,64),direct=run(1,0,64),graph=run(2,0,64),republished=run(2,0,64,true);
    assert(std::set<double>(legacy.y.begin(),legacy.y.end()).size()==1);
    assert(std::set<double>(direct.y.begin(),direct.y.end()).size()==64);
    assert(direct.y==graph.y&&direct.z==graph.z&&direct.y==republished.y&&direct.z==republished.z);
    assert(legacy.diameter==direct.diameter&&legacy.rng==direct.rng&&direct.rng==graph.rng);
    const auto first=run(2,0,32),second=run(2,32,32);
    for(int i=0;i<32;++i){assert(first.y[i]==graph.y[i]&&first.z[i]==graph.z[i]);assert(second.y[i]==graph.y[i+32]&&second.z[i]==graph.z[i+32]);}
    const auto high=run(2,(1ULL<<53)+17,64),highDirect=run(1,(1ULL<<53)+17,64);
    assert(high.y==highDirect.y&&high.z==highDirect.z);
    std::printf("PASS CUDA actual injection: fixed boundary=.5, dt=.5, mass=64, legacy_unique=1, sequence_unique=64, direct=graph=republished, restart32+32=continuous64, high_uint64; graph recordings=%d parameter_updates=%d\n",graph.recordings,graph.updates);
}
#endif
int main(int argc,char** argv){
    cpuChecks(argc>1?argv[1]:"/tmp/ugkp-inlet-sequence-test.dat");
#ifdef __CUDACC__
    gpuChecks();
#endif
}
