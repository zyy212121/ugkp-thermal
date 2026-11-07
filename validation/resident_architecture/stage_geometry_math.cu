#include "common/gpu/StageGeometryMath.H"
#include <cuda_runtime.h>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
using namespace ugkp::gpu::geometry;
struct Input {double volume[4];Point old,end,area[6];double sweep[6];int axis;};
struct Output {bool inverts;Point sample[3];StageCellConservation conservation;};
// Frozen native Geometry.C formulas at 573005d2, independently evaluated.
Output reference(const Input& in){
    Output out{};const double v0=in.volume[0],v1=in.volume[1],v2=in.volume[2],v3=in.volume[3];
    const double a=4.5*(v3-3*v2+3*v1-v0),b=4.5*(v2-2*v1+v0)-a,d=3*(v1-v0)-a/9-b/3;
    const double small=256*std::numeric_limits<double>::epsilon()*(v0>v3?v0:v3);
    double critical[2]={-1,-1};
    if(std::abs(a)<=small){if(std::abs(b)>small)critical[0]=-d/(2*b);}
    else{const double disc=4*b*b-12*a*d;if(disc>=0){const double root=::sqrt(disc);critical[0]=(-2*b-root)/(6*a);critical[1]=(-2*b+root)/(6*a);}}
    for(double t:critical)if(t>0&&t<1&&((a*t+b)*t+d)*t+v0<=0)out.inverts=true;
    out.sample[0]=in.axis<0?(in.old+in.end)*.5:in.old+(in.end-in.old)*.5;
    out.sample[1]=in.axis<0?in.old*(2.0/3)+in.end/3:in.old+(in.end-in.old)/3;
    out.sample[2]=in.axis<0?in.old/3+in.end*(2.0/3):in.old+(in.end-in.old)*(2.0/3);
    double sum=0,area=0;Point closure{};
    for(int k=0;k<6;++k){const int sign=k%2?1:-1;sum+=sign*in.sweep[k];closure+=in.area[k]*sign;area+=magnitude(in.area[k]);}
    auto& r=out.conservation;r.residualLimit=1e-14+1e-10*std::max(v0,v3);
    r.invalid=!finite(closure)||!std::isfinite(area);
    if(!r.invalid){r.closure=magnitude(closure);r.closureLimit=1e-10*area;r.residual=v3-v0-sum;}
    return out;
}
struct Area {const Point* p;__host__ __device__ Point operator()(int i)const{return p[i];}};
struct Sweep {const double* p;__host__ __device__ double operator()(int i)const{return p[i];}};
__global__ void evaluate(int count,const Input* input,Output* output){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;const auto& in=input[i];auto& out=output[i];
    out.inverts=stageInteriorVolumeInverts(in.volume[0],in.volume[1],in.volume[2],in.volume[3]);
    for(int sample=0;sample<3;++sample)out.sample[sample]=stageSample(in.old,in.end,sample,in.axis);
    const int offsets[]={0,6},faces[]={0,1,2,3,4,5},signs[]={-1,1,-1,1,-1,1};
    const TopologyView topology{nullptr,nullptr,offsets,faces,signs,nullptr,nullptr};
    out.conservation=stageCellConservation(0,topology,in.volume[0],in.volume[3],Area{in.area},Sweep{in.sweep},1e-14,1e-10);
}
void check(cudaError_t e){if(e!=cudaSuccess){std::cerr<<cudaGetErrorString(e)<<'\n';std::abort();}}
void exact(double a,double b){if(std::isnan(a)&&std::isnan(b))return;if(std::memcmp(&a,&b,sizeof(a))){std::cerr.precision(17);std::cerr<<"different "<<a<<' '<<b<<'\n';std::abort();}}
int main(){
    std::mt19937_64 random(51120);std::uniform_real_distribution<double> value(-1.,1.);std::vector<Input> input;
    for(int exponent:{-900,-600,-60,0,60,600,900})for(int sample=0;sample<4096;++sample){Input in{};in.axis=sample%4-1;
        for(auto& v:in.volume)v=std::ldexp(std::abs(value(random)),exponent);
        in.old={std::ldexp(value(random),exponent/3),std::ldexp(value(random),exponent/3),std::ldexp(value(random),exponent/3)};
        in.end=sample%3?in.old+Point{std::ldexp(value(random),exponent/3),std::ldexp(value(random),exponent/3),std::ldexp(value(random),exponent/3)}:in.old;
        for(int f=0;f<6;++f){in.area[f]={std::ldexp(value(random),exponent/3),std::ldexp(value(random),exponent/3),std::ldexp(value(random),exponent/3)};in.sweep[f]=std::ldexp(value(random),exponent);}
        input.push_back(in);
    }
    Input* deviceInput=nullptr;Output* deviceOutput=nullptr;check(cudaMalloc(&deviceInput,input.size()*sizeof(Input)));check(cudaMalloc(&deviceOutput,input.size()*sizeof(Output)));
    check(cudaMemcpy(deviceInput,input.data(),input.size()*sizeof(Input),cudaMemcpyHostToDevice));
    evaluate<<<(input.size()+127)/128,128>>>(input.size(),deviceInput,deviceOutput);check(cudaGetLastError());
    std::vector<Output> output(input.size());check(cudaMemcpy(output.data(),deviceOutput,output.size()*sizeof(Output),cudaMemcpyDeviceToHost));
    int inversions=0;
    for(std::size_t i=0;i<input.size();++i){const auto expected=reference(input[i]);const auto& actual=output[i];
        assert(actual.inverts==expected.inverts);inversions+=actual.inverts;
        for(int k=0;k<3;++k){exact(actual.sample[k].x,expected.sample[k].x);exact(actual.sample[k].y,expected.sample[k].y);exact(actual.sample[k].z,expected.sample[k].z);}
        const auto& a=actual.conservation;const auto& b=expected.conservation;assert(a.invalid==b.invalid);
        exact(a.closure,b.closure);exact(a.closureLimit,b.closureLimit);exact(a.residual,b.residual);exact(a.residualLimit,b.residualLimit);
    }
    check(cudaFree(deviceInput));check(cudaFree(deviceOutput));
    std::cout<<"PASS "<<input.size()<<" FP64 stage samples, volume extrema and GCL balances; "<<inversions<<" interior inversions; host/device results bitwise equal\n";return 0;
}
