#include "GpuBatchedMaximum.cuh"
#include "gasTransport/GasPseudoTime.H"
#include <cstdio>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <cstdint>
struct State {
    ugkwp::GasResidentInterval gasResidentInterval;
    const double* gasPseudoDt=nullptr;
    double gasPseudoReferenceDt=1;
};
__global__ void evaluateInterval(State state,double* output){
    if(threadIdx.x||blockIdx.x)return;
    output[0]=ugkwp::gasCellTimeStep(state,0,1.0);
    output[1]=ugkwp::gasCellTimeStep(state,0,1.0/6.0);
    output[2]=ugkwp::gasResidentStageTime(state,0.0);
    output[3]=ugkwp::gasResidentStageTime(state,1.0);
    output[4]=ugkwp::gasResidentStageTime(state,0.5);
}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
bool equal(double a,double b){return std::isnan(a)?std::isnan(b):a==b;}
int main(){
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for(int count:{0,1,255,256,257,4592,8193})for(int components:{1,2,9}){
        std::vector<double> values(size_t(count)*components);
        for(size_t i=0;i<values.size();++i)values[i]=double((i*731)%9967)-5000;
        if(count&&components>1)values[size_t(count)+count/2]=NAN;
        if(count&&components>2)values[size_t(count)*2]=INFINITY;
        const int parts=ugkwp::componentMaximumParts(count);
        double *device=nullptr,*workspace=nullptr;
        check(cudaMalloc(&device,(values.empty()?1:values.size())*sizeof(double)));
        check(cudaMalloc(&workspace,(size_t(parts+1)*components+1)*sizeof(double)));
        const double guard=12345.125;
        check(cudaMemcpyAsync(device,values.data(),values.size()*sizeof(double),cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(workspace+size_t(parts+1)*components,&guard,sizeof(double),cudaMemcpyHostToDevice,stream));
        check(ugkwp::batchedComponentMaximum(device,count,components,workspace,stream));
        std::vector<double> result(components+1);
        check(cudaMemcpyAsync(result.data(),workspace+size_t(parts)*components,result.size()*sizeof(double),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));
        for(int k=0;k<components;++k){double expected=-INFINITY;
            for(int i=0;i<count;++i){const double v=values[size_t(k)*count+i];expected=std::isnan(expected)?expected:std::isnan(v)?v:std::fmax(expected,v);}
            if(!equal(expected,result[k]))return 2;}
        if(result.back()!=guard)return 3;
        check(cudaFree(device));check(cudaFree(workspace));
    }
    for(int count:{0,1,255,256,257,4592,8193})for(int components:{1,5,60}){
        const int maxima=2*components/5,parts=ugkwp::componentMaximumParts(count);
        std::vector<double> values(size_t(count)*components);
        for(size_t i=0;i<values.size();++i)
            values[i]=std::ldexp(double(int((i*731)%9967)-5000)/7.0,int(i%17)-8);
        double *input=nullptr,*scratch=nullptr;const double guard=9876.125;
        check(cudaMalloc(&input,(values.empty()?1:values.size())*sizeof(double)));
        check(cudaMalloc(&scratch,(size_t(parts+1)*components+1)*sizeof(double)));
        check(cudaMemcpyAsync(input,values.data(),values.size()*sizeof(double),cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(scratch+size_t(parts+1)*components,&guard,sizeof(double),cudaMemcpyHostToDevice,stream));
        check(ugkwp::batchedComponentStatistics(input,count,components,maxima,scratch,stream));
        std::vector<double> result(components+1);
        check(cudaMemcpyAsync(result.data(),scratch+size_t(parts)*components,result.size()*sizeof(double),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));
        for(int k=0;k<components;++k){
            double maximum=-INFINITY;long double sum=0,absolute=0;
            for(int i=0;i<count;++i){double v=values[size_t(k)*count+i];
                maximum=std::fmax(maximum,v);sum+=v;absolute+=std::abs(v);}
            if(k<maxima){if(!equal(maximum,result[k]))return 5;}
            else if(!std::isfinite(result[k])||std::abs(static_cast<long double>(result[k])-sum)
                >128*std::numeric_limits<double>::epsilon()*absolute)return 6;
        }
        if(result.back()!=guard)return 7;
        check(cudaFree(input));check(cudaFree(scratch));
    }
    double* device;check(cudaMalloc(&device,6*sizeof(double)));
    for(double dt:{1e-50,1e-8,0.003,1.0,1e10})for(double time:{0.0,1e-6,3.0,1e20})for(bool frozen:{false,true})for(bool pseudo:{false,true}){
        State state;state.gasResidentInterval={dt,time,time+dt,time+dt*0.5,true,frozen};
        double local=0.013,result[5];
        if(pseudo){check(cudaMemcpyAsync(device+5,&local,sizeof(double),cudaMemcpyHostToDevice,stream));state.gasPseudoDt=device+5;}
        evaluateInterval<<<1,1,0,stream>>>(state,device);check(cudaGetLastError());
        check(cudaMemcpyAsync(result,device,sizeof(result),cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
        const double expected[5]={pseudo?dt*local:dt,pseudo?(dt*(1.0/6.0))*local:dt*(1.0/6.0),time,frozen?time:time+dt,frozen?time:time+dt*0.5};
        for(int k=0;k<5;++k)if(std::memcmp(result+k,expected+k,sizeof(double)))return 4;
    }
    check(cudaFree(device));check(cudaStreamDestroy(stream));
    std::puts("PASS 21 maximum and 21 mixed-statistic reductions; 80 FP64 resident interval combinations");
}
