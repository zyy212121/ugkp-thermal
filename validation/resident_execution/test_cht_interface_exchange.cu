#include "../../applications/CHT/gpu/GpuWallEnergy64.H"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

extern "C" int ugkpInvalidateGasWallEnergyGraph(void*) { return 0; }
extern "C" int ugkpPrepareParticleWallEnergyRestore(void*) { return 1; }
extern "C" int ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(void*,int,const int*);
extern "C" int ugkwpGpuResidentStrictConfigureInterfaceExchange(void*,int,const int*,const int*,const double*);
extern "C" int ugkwpGpuResidentStrictPreviewInterfaceExchange(void*,double,const double*,double*);
extern "C" int ugkwpGpuResidentStrictPreviewInterfaceExchangeDevice(void*,double,const ugkp::gpu::RadiationMaterialDeviceView*,double*,ugkp::gpu::ThermalInterfaceEnergyDeviceView*);
extern "C" int ugkwpGpuResidentStrictUploadWallEnergyLedgerRange(void*,int,int,const double*,const double*,const double*);
extern "C" int ugkwpGpuResidentStrictPeekWallEnergyLedgerRange(void*,int,int,double*,double*,double*);

static void cudaCheck(cudaError_t e) { if(e!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::abort(); } }
static void check(int e,const GpuWallEnergyState& s) { if(e) { std::fprintf(stderr,"%s\n",s.errorBuffer);std::abort(); } }
template<class T> static void allocate(T*& p,size_t n) { cudaCheck(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T))); }
static void equal(double actual,double expected)
{
    if(actual!=expected) { std::fprintf(stderr,"actual %.17g expected %.17g\n",actual,expected);std::abort(); }
}

// The oracle below is the removed host exchange: gas + radiation +
// (deposited + reflected), followed by the original fluid-face ordered map.
int main(int argc,char** argv)
{
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices==0)return 77;
    const bool benchmark=argc>1&&std::strcmp(argv[1],"--benchmark")==0;
    constexpr int meshFaces=262144, count=4096, rows=2305;
    constexpr double dt=0.03125;
    std::vector<int> faces(count), destination(count), offsets(rows+1,0);
    std::vector<double> areas(count),gas(meshFaces),dep(meshFaces),ref(meshFaces),rad(count);
    // Two disjoint patches; both map nonconformally with unequal row degrees.
    for(int i=0;i<count;++i)
    {
        faces[i]=(i<count/2?65536+i:196608+i-count/2);
        destination[i]=i<count/2?i%1100:1100+(i-count/2)%1204;
        ++offsets[destination[i]+1];
        areas[i]=0.125+double(i%23)/32;
        gas[faces[i]]=double(i%29-14)*0.125;
        dep[faces[i]]=double(i%7-3)*0.03125;
        ref[faces[i]]=double(i%11-5)*0.0625;
        rad[i]=double(i%17-8)*0.015625;
    }
    // Cancellation cases detect atomics or reordered sums.
    gas[faces[0]]=1e16;gas[faces[1100]]=-1e16;
    for(int row=0;row<rows;++row)offsets[row+1]+=offsets[row];
    std::vector<int> cursor(offsets),sources(count);
    for(int i=0;i<count;++i)sources[cursor[destination[i]]++]=i;

    char error[2048]={};
    GpuWallEnergyState s;
    s.nFaces=meshFaces;s.errorBuffer=error;s.nInternalFaces=65536;
    allocate(s.deviceState,1);
    allocate(s.particleWallDepositedEnergy,meshFaces);
    allocate(s.particleWallReflectedEnergy,meshFaces);
    check(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&s,count,faces.data()),s);
    cudaCheck(cudaMemcpy(s.gasWallEnergy,gas.data(),meshFaces*sizeof(double),cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(s.particleWallDepositedEnergy,dep.data(),meshFaces*sizeof(double),cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(s.particleWallReflectedEnergy,ref.data(),meshFaces*sizeof(double),cudaMemcpyHostToDevice));
    // Replay one captured RK ledger graph with varying resident intervals.
    // frozen affects stage-time selection, never the integrated heat interval;
    // dt must multiply the RK weight exactly once, including after graph reuse.
    std::vector<double> flux(meshFaces,0),ledger(meshFaces),ledgerExpected;
    for(int i=0;i<count;++i)flux[faces[i]]=double(i%31-15)*0.3125;
    cudaCheck(cudaMemcpy(s.gasWallFlux,flux.data(),meshFaces*sizeof(double),cudaMemcpyHostToDevice));
    ugkwp::GasResidentInterval* interval=nullptr;allocate(interval,1);
    cudaStream_t ledgerStream=nullptr;cudaCheck(cudaStreamCreateWithFlags(&ledgerStream,cudaStreamNonBlocking));
    cudaGraph_t ledgerGraph=nullptr;cudaGraphExec_t ledgerExec=nullptr;
    const double weights[]={1.0/6,2.0/3,1.0/6};
    cudaCheck(cudaStreamBeginCapture(ledgerStream,cudaStreamCaptureModeThreadLocal));
    for(double weight:weights)check(ugkpAccumulateGasWallEnergy64(&s,weight,ledgerStream,interval),s);
    cudaCheck(cudaStreamEndCapture(ledgerStream,&ledgerGraph));
    cudaCheck(cudaGraphInstantiateWithFlags(&ledgerExec,ledgerGraph,0));
    int intervalChecks=0;
    for(bool active:{false,true})for(bool frozen:{false,true})for(double dt:{0.0,0x1p-40,0.03125,16.75}){
        const ugkwp::GasResidentInterval value{dt,2.0,2.0+dt,2.0+0.5*dt,active,frozen};
        cudaCheck(cudaMemcpy(s.gasWallEnergy,gas.data(),meshFaces*sizeof(double),cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpyAsync(interval,&value,sizeof value,cudaMemcpyHostToDevice,ledgerStream));
        cudaCheck(cudaGraphLaunch(ledgerExec,ledgerStream));cudaCheck(cudaStreamSynchronize(ledgerStream));
        cudaCheck(cudaMemcpy(ledger.data(),s.gasWallEnergy,meshFaces*sizeof(double),cudaMemcpyDeviceToHost));
        ledgerExpected=gas;
        for(double weight:weights)for(int face:faces)ledgerExpected[face]+=(active?weight*dt:weight)*flux[face];
        for(int face=0;face<meshFaces;++face)equal(ledger[face],ledgerExpected[face]);
        ++intervalChecks;
    }
    // The original explicit-time API remains identical when no resident
    // interval is supplied.
    cudaCheck(cudaMemcpy(s.gasWallEnergy,gas.data(),meshFaces*sizeof(double),cudaMemcpyHostToDevice));
    check(ugkpAccumulateGasWallEnergy64(&s,0.125,ledgerStream),s);cudaCheck(cudaStreamSynchronize(ledgerStream));
    cudaCheck(cudaMemcpy(ledger.data(),s.gasWallEnergy,meshFaces*sizeof(double),cudaMemcpyDeviceToHost));
    ledgerExpected=gas;for(int face:faces)ledgerExpected[face]+=0.125*flux[face];
    for(int face=0;face<meshFaces;++face)equal(ledger[face],ledgerExpected[face]);
    cudaCheck(cudaMemcpy(s.gasWallEnergy,gas.data(),meshFaces*sizeof(double),cudaMemcpyHostToDevice));
    cudaCheck(cudaGraphExecDestroy(ledgerExec));cudaCheck(cudaGraphDestroy(ledgerGraph));
    cudaCheck(cudaStreamDestroy(ledgerStream));cudaCheck(cudaFree(interval));
    std::printf("Wall ledger interval graph: %d active/frozen/dt combinations and explicit-time API PASS\n",intervalChecks);
    std::vector<double> actual(3*count+rows),expected(actual.size());
    cudaStream_t radiationStream=nullptr;cudaCheck(cudaStreamCreateWithFlags(&radiationStream,cudaStreamNonBlocking));
    const int boundaryFaces=meshFaces-s.nInternalFaces;
    std::vector<double> boundaryRadiation(boundaryFaces,-12345),deviceEnergy(rows);
    for(int i=0;i<count;++i)boundaryRadiation[faces[i]-s.nInternalFaces]=rad[i];
    double* deviceRadiation=nullptr;allocate(deviceRadiation,boundaryFaces);
    ugkp::gpu::RadiationMaterialDeviceView radiationView;
    radiationView.enabled=true;radiationView.boundaryFaces=boundaryFaces;
    radiationView.boundaryEnergy=deviceRadiation;radiationView.producerStream=radiationStream;
    for(int particles=0;particles<2;++particles)
    {
        s.particleWallHeatTransferEnabled=particles;
        check(ugkwpGpuResidentStrictConfigureInterfaceExchange(&s,rows,offsets.data(),sources.data(),areas.data()),s);
        for(int radiation=0;radiation<2;++radiation)
        {
            std::fill(expected.begin(),expected.end(),0);
            for(int i=0;i<count;++i)
            {
                const int face=faces[i];
                double combined=gas[face];
                if(radiation)combined+=rad[i];
                if(particles)combined+=dep[face]+ref[face];
                expected[3*count+destination[i]]+=combined;
                expected[i]=gas[face]/(areas[i]*dt);
                expected[count+i]=(particles?dep[face]:0)/(areas[i]*dt);
                expected[2*count+i]=(particles?ref[face]:0)/(areas[i]*dt);
            }
            check(ugkwpGpuResidentStrictPreviewInterfaceExchange(&s,dt,radiation?rad.data():nullptr,actual.data()),s);
            for(size_t i=0;i<actual.size();++i)equal(actual[i],expected[i]);
            // The resident path consumes a different stream's complete boundary
            // array, then exposes exactly the same compact solid-face energies.
            cudaCheck(cudaMemcpyAsync(deviceRadiation,boundaryRadiation.data(),boundaryFaces*sizeof(double),cudaMemcpyHostToDevice,radiationStream));
            ugkp::gpu::ThermalInterfaceEnergyDeviceView retained;
            check(ugkwpGpuResidentStrictPreviewInterfaceExchangeDevice(&s,dt,radiation?&radiationView:nullptr,actual.data(),&retained),s);
            assert(retained.enabled&&retained.faces==rows&&retained.energy);
            for(size_t i=0;i<actual.size();++i)equal(actual[i],expected[i]);
            cudaCheck(cudaMemcpy(deviceEnergy.data(),retained.energy,rows*sizeof(double),cudaMemcpyDeviceToHost));
            for(int i=0;i<rows;++i)equal(deviceEnergy[i],expected[3*count+i]);
            // No host preview is required to publish the physical device view.
            check(ugkwpGpuResidentStrictPreviewInterfaceExchangeDevice(&s,dt,radiation?&radiationView:nullptr,nullptr,&retained),s);
            cudaCheck(cudaMemcpy(deviceEnergy.data(),retained.energy,rows*sizeof(double),cudaMemcpyDeviceToHost));
            for(int i=0;i<rows;++i)equal(deviceEnergy[i],expected[3*count+i]);
            // Preview is non-destructive and repeatable; pending/restart data
            // remains the exact input ledger, rather than an exchanged total.
            check(ugkwpGpuResidentStrictPreviewInterfaceExchange(&s,dt,radiation?rad.data():nullptr,actual.data()),s);
            for(size_t i=0;i<actual.size();++i)equal(actual[i],expected[i]);
            std::vector<double> g(count/2),d(count/2),r(count/2);
            check(ugkwpGpuResidentStrictPeekWallEnergyLedgerRange(&s,faces[0],count/2,g.data(),d.data(),r.data()),s);
            for(int i=0;i<count/2;++i)
            {
                equal(g[i],gas[faces[i]]);
                equal(d[i],particles?dep[faces[i]]:0);
                equal(r[i],particles?ref[faces[i]]:0);
            }
        }
    }
    // The existing restart API modifies resident ledgers used by the next
    // preview, without rebuilding the persisted interface map.
    double restoredGas=7,restoredDep=-2,restoredRef=0.5;
    check(ugkwpGpuResidentStrictUploadWallEnergyLedgerRange(&s,faces[7],1,&restoredGas,&restoredDep,&restoredRef),s);
    gas[faces[7]]=restoredGas;dep[faces[7]]=restoredDep;ref[faces[7]]=restoredRef;
    check(ugkwpGpuResidentStrictPreviewInterfaceExchange(&s,dt,nullptr,actual.data()),s);
    equal(actual[7],restoredGas/(areas[7]*dt));
    equal(actual[count+7],restoredDep/(areas[7]*dt));
    equal(actual[2*count+7],restoredRef/(areas[7]*dt));
    double rowExpected=0;
    for(int i=0;i<count;++i)if(destination[i]==destination[7])
        rowExpected+=gas[faces[i]]+(dep[faces[i]]+ref[faces[i]]);
    equal(actual[3*count+destination[7]],rowExpected);

    if(benchmark)
    {
        constexpr int repeats=100;
        std::vector<double> hostGas(meshFaces),hostDep(meshFaces),hostRef(meshFaces),hostCombined(count),hostSolid(rows);
        auto start=std::chrono::steady_clock::now();
        for(int repeat=0;repeat<repeats;++repeat)
        {
            cudaCheck(cudaMemcpy(hostGas.data(),s.gasWallEnergy,meshFaces*sizeof(double),cudaMemcpyDeviceToHost));
            cudaCheck(cudaMemcpy(hostDep.data(),s.particleWallDepositedEnergy,meshFaces*sizeof(double),cudaMemcpyDeviceToHost));
            cudaCheck(cudaMemcpy(hostRef.data(),s.particleWallReflectedEnergy,meshFaces*sizeof(double),cudaMemcpyDeviceToHost));
            std::fill(hostSolid.begin(),hostSolid.end(),0);
            for(int i=0;i<count;++i)
            {
                const int face=faces[i];
                hostCombined[i]=hostGas[face]+(hostDep[face]+hostRef[face]);
                hostSolid[destination[i]]+=hostCombined[i];
                expected[i]=hostGas[face]/(areas[i]*dt);
                expected[count+i]=hostDep[face]/(areas[i]*dt);
                expected[2*count+i]=hostRef[face]/(areas[i]*dt);
            }
        }
        const auto hostEnd=std::chrono::steady_clock::now();
        for(int repeat=0;repeat<repeats;++repeat)
            check(ugkwpGpuResidentStrictPreviewInterfaceExchange(&s,dt,nullptr,actual.data()),s);
        const auto gpuEnd=std::chrono::steady_clock::now();
        const double oldMs=std::chrono::duration<double,std::milli>(hostEnd-start).count()/repeats;
        const double newMs=std::chrono::duration<double,std::milli>(gpuEnd-hostEnd).count()/repeats;
        std::printf("mesh_faces=%d interface_faces=%d solid_faces=%d old_ms=%.6f gpu_ms=%.6f speedup=%.3f old_d2h_bytes=%zu gpu_d2h_bytes=%zu\n",
                    meshFaces,count,rows,oldMs,newMs,oldMs/newMs,3*gas.size()*sizeof(double),actual.size()*sizeof(double));
        for(int row=0;row<rows;++row)equal(actual[3*count+row],hostSolid[row]);
    }
    auto invalidRadiation=radiationView;invalidRadiation.boundaryFaces-=1;
    ugkp::gpu::ThermalInterfaceEnergyDeviceView invalidOutput;
    assert(ugkwpGpuResidentStrictPreviewInterfaceExchangeDevice(&s,dt,&invalidRadiation,nullptr,&invalidOutput)!=0);
    assert(!invalidOutput.enabled);
    cudaCheck(cudaStreamSynchronize(radiationStream));cudaCheck(cudaStreamDestroy(radiationStream));cudaCheck(cudaFree(deviceRadiation));
    // Empty local interface is a legitimate partition and never dereferences a
    // null allocation or launches a zero-grid kernel.
    GpuWallEnergyState empty;empty.errorBuffer=error;allocate(empty.deviceState,1);
    const int emptyOffset=0;check(ugkwpGpuResidentStrictConfigureInterfaceExchange(&empty,0,&emptyOffset,nullptr,nullptr),empty);
    ugkp::gpu::ThermalInterfaceEnergyDeviceView emptyOutput;
    check(ugkwpGpuResidentStrictPreviewInterfaceExchangeDevice(&empty,dt,nullptr,nullptr,&emptyOutput),empty);
    assert(emptyOutput.enabled&&emptyOutput.faces==0&&!emptyOutput.energy);
    ugkpReleaseGasWallEnergy64(&empty);cudaCheck(cudaFree(empty.deviceState));
    ugkpReleaseGasWallEnergy64(&s);
    cudaCheck(cudaFree(s.particleWallDepositedEnergy));
    cudaCheck(cudaFree(s.particleWallReflectedEnergy));
    cudaCheck(cudaFree(s.deviceState));
    std::puts("CHT interface exchange CPU/GPU oracle, conservation mapping, pending ledger and restore: PASS");
}
