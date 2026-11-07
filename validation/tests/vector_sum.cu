// nvcc -std=c++17 -O2 --fmad=false -Icommon validation/tests/vector_sum.cu -o vector-sum
#include "GpuVectorSum.cuh"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
static void check(cudaError_t error){if(error!=cudaSuccess){
    std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(error));std::exit(1);}}
struct Source {
    const double* values;const int* active;int stride;
    __device__ int count()const{return *active;}
    __device__ void add(int i,double* q)const{
        // Sparse active entries, signed moment cancellation, multiple scales.
        if(i%7==0)return;
        for(int k=0;k<6;++k)q[k]+=values[size_t(k)*stride+i];
    }
};
__global__ void atomicReference(Source source,double* output,double sign){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<source.count();i+=gridDim.x*blockDim.x){
        double q[6]={};source.add(i,q);
        for(int k=0;k<6;++k)atomicAdd(output+k,sign*q[k]);
    }
}
int main(int argc,char** argv){
    const int capacity=262147;
    std::vector<double> input(size_t(6)*capacity);
    for(int k=0;k<6;++k)for(int i=0;i<capacity;++i)
        input[size_t(k)*capacity+i]=std::ldexp(double((i*17+k*19)%127-63),k*7-20);
    double *values,*workspace,*output;int* count;
    check(cudaMalloc(&values,input.size()*sizeof(double)));
    check(cudaMalloc(&output,6*sizeof(double)));check(cudaMalloc(&count,sizeof(int)));
    check(cudaMemcpy(values,input.data(),input.size()*sizeof(double),cudaMemcpyHostToDevice));
    int device,sm,blocks;check(cudaGetDevice(&device));
    check(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,device));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,
        ugkwp::vectorSumPartials<6,256,Source>,256,0));
    const int parts=sm*blocks;check(cudaMalloc(&workspace,size_t(parts)*6*sizeof(double)));
    cudaStream_t stream;check(cudaStreamCreate(&stream));Source source{values,count,capacity};
    for(int n:{0,1,31,32,33,255,256,257,8193,capacity})for(int split:{1,3,parts}){
        check(cudaMemcpyAsync(count,&n,sizeof(int),cudaMemcpyHostToDevice,stream));
        check(cudaMemsetAsync(output,0,6*sizeof(double),stream));
        check(ugkwp::accumulateVectorSum<6>(source,split,workspace,output,1.,stream));
        check(ugkwp::accumulateVectorSum<6>(source,split,workspace,output,-.25,stream));
        double result[6];check(cudaMemcpyAsync(result,output,sizeof(result),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));
        for(int k=0;k<6;++k){long double reference=0.,magnitude=0.;
            for(int i=0;i<n;++i)if(i%7){reference+=input[size_t(k)*capacity+i];magnitude+=std::abs(input[size_t(k)*capacity+i]);}
            const double error=std::abs(result[k]-double(.75L*reference));
            if(error>2e-14*double(magnitude)+1e-30){std::fprintf(stderr,"sum mismatch %d %d %d %.17g\n",n,split,k,error);return 2;}
        }
    }
    // Replay uses the current device count, including a later enlarged inventory.
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    check(cudaMemsetAsync(output,0,6*sizeof(double),stream));
    check(ugkwp::accumulateVectorSum<6>(source,parts,workspace,output,1.,stream));
    cudaGraph_t graph;cudaGraphExec_t executable;
    check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiateWithFlags(&executable,graph,0));
    for(int n:{0,33,capacity}){
        check(cudaMemcpyAsync(count,&n,sizeof(int),cudaMemcpyHostToDevice,stream));
        check(cudaGraphLaunch(executable,stream));
        double result[6];check(cudaMemcpyAsync(result,output,sizeof(result),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));
        for(int k=0;k<6;++k){long double reference=0.;for(int i=0;i<n;++i)if(i%7)reference+=input[size_t(k)*capacity+i];
            assert(std::abs(result[k]-double(reference))<=1e-12*std::abs(double(reference))+1e-25);}
    }
    if(argc>1&&std::string(argv[1])=="--benchmark"){
        const int n=capacity;check(cudaMemcpy(count,&n,sizeof(int),cudaMemcpyHostToDevice));
        cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
        for(int variant=0;variant<2;++variant){
            for(int j=0;j<4;++j){
                check(cudaMemsetAsync(output,0,6*sizeof(double),stream));
                check(cudaEventRecord(begin,stream));
                for(int repeat=0;repeat<100;++repeat){
                    if(!variant)atomicReference<<<parts,256,0,stream>>>(source,output,1.);
                    else check(ugkwp::accumulateVectorSum<6>(source,parts,workspace,output,1.,stream));
                }
                check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
                float ms;check(cudaEventElapsedTime(&ms,begin,end));
                std::printf("audit variant=%d repetition=%d us=%.6f\n",variant,j,ms*10.);
            }
        }
        check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
    }
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
    check(cudaStreamDestroy(stream));check(cudaFree(values));check(cudaFree(workspace));check(cudaFree(output));check(cudaFree(count));
    std::puts("vector-sum: 30 signed sparse cases and 3 captured count changes passed");
}
