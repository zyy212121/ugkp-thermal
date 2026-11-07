#include "gpu/ParticleRadiationMaterial.H"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace ugkp::gpu;
static void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}
template<class T> struct Array {
    T* data=nullptr;
    explicit Array(std::vector<T> x){check(cudaMalloc(&data,std::max(std::size_t(1),x.size())*sizeof(T)));if(!x.empty())check(cudaMemcpy(data,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));}
    ~Array(){cudaFree(data);}
};
__global__ void publishResponse(ParticleRadiationResponseView* output,ParticleRadiationResponseView a,
    ParticleRadiationResponseView b,int phase){if(!blockIdx.x&&!threadIdx.x)*output=phase%2?b:a;}
static MieOpticalTable table(){MieOpticalTable t;t.temperature={100,2000};t.diameter={1e-4,1e-2};t.mu={-1,1};t.band={1,1};
    t.absorption.assign(4,1);t.scattering.assign(4,.3);t.extinction.assign(4,1.3);t.phase.assign(8,1);return t;}
int main(){
    Array<int> count({6}),cell({0,0,0,1,1,1}),status({1,1,1,1,1,1}),offset({0,3,6,6,6});
    Array<unsigned char> free({0,0,0,0,0,0}),contact({0,1,0,0,0,0});
    Array<GpuReal> mass({1.,2.,3.,1.,2.,3.}),ta({500.,600.,700.,650.,700.,750.}),tb({900.,800.,700.,750.,700.,650.});
    Array<double> carry({1e-9,0.,-1e-9,0.,0.,0.}),epsilon({.001,.001,0.,0.}),bulkMass({6.,6.,0.,0.}),bulkT({650.,700.,300.,300.}),diameter({.001,.001,.001,.001}),incident({1e5,2e5,0.,0.}),volume({1.,1.,1.,1.});
    ParticleRadiationResponseView a{true,6,count.data,cell.data,status.data,offset.data,free.data,mass.data,ta.data,carry.data,100.,2000.};
    auto b=a;b.temperature=tb.data;b.wallState=contact.data;
    Array<ParticleRadiationResponseView> descriptor({a});
    ParticleRadiationMaterial candidate(table()),reference(table());ParticleRadiationMaterialInput input;
    input.heatFactor=1000;input.elapsed=.01;input.minimumTemperature=100;input.maximumTemperature=2000;
    cudaStream_t producer;check(cudaStreamCreateWithFlags(&producer,cudaStreamNonBlocking));
    input.resident={true,2,0,epsilon.data,bulkMass.data,bulkT.data,diameter.data,incident.data,nullptr,volume.data,nullptr,nullptr,producer};
    int failures=0;double maximum=0;
    for(int phase=0;phase<12;++phase){
        // Direct sorted -> unsorted -> sorted, with buffer swaps and attached
        // particles. A larger direct caller must invalidate cached scratch too.
        const bool sorted=phase%3!=1;auto aa=a,bb=b;if(!sorted)aa.cellOffsets=bb.cellOffsets=nullptr;
        publishResponse<<<1,1,0,producer>>>(descriptor.data,aa,bb,phase);
        input.resident.cells=phase==6?4:2;
        input.response=phase%2?bb:aa;
        const auto expected=reference.solve(input);
        if(phase==6){input.response.cellOffsets=nullptr;candidate.solve(input);}
        input.response={};input.response.enabled=true;input.response.capacity=6;input.response.deviceView=descriptor.data;
        const auto actual=candidate.solve(input);
        for(int c=0;c<input.resident.cells;++c)for(int field=0;field<2;++field){
            const double x=field?actual.energy[c]:actual.temperature[c],y=field?expected.energy[c]:expected.temperature[c];
            const double error=std::abs(x-y)/std::max(1.,std::abs(y));maximum=std::max(maximum,error);
            if(!std::isfinite(x)||error>2e-12)++failures;
        }
    }
    check(cudaStreamDestroy(producer));
    std::printf("radiation descriptor: twelve swaps/directory/growth calls, maximum scaled error %.17g, %d failures\n",maximum,failures);
    return failures?1:0;
}
