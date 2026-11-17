#include "../../common/gpu/ScalarWorkLaunch.cuh"
#include <cstdio>
#include <vector>
__global__ void markWorkers(int workers,int active,int* counts) {
    if(int(threadIdx.x)>=active)return;
    int worker=int(blockIdx.x)*active+int(threadIdx.x);
    if(worker<workers)atomicAdd(counts+worker,1);
}
int main(){
    for(int b:{32,96,128,256})for(int n:{0,1,2,7,31,32,127,128,129,513,10000}){
        ugkp::gpu::workPlan().materialBlockThreads=b;
        ugkp::gpu::ScalarWorkLaunch q;
        auto err=ugkp::gpu::scalarWorkLaunch(markWorkers,n,q);
        if(err!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(err));return 2;}
        if(!n){if(q.blocks)return 3;continue;}
        if(q.threads!=b||q.workersPerBlock<1||q.workersPerBlock>b)return 4;
        int* d=nullptr; if(cudaMalloc(&d,n*sizeof(int))!=cudaSuccess)return 5;
        cudaMemset(d,0,n*sizeof(int));
        markWorkers<<<q.blocks,q.threads>>>(n,q.workersPerBlock,d);
        std::vector<int> result(n);
        if(cudaMemcpy(result.data(),d,n*sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess)return 6;
        cudaFree(d);
        for(int x:result)if(x!=1)return 7;
        std::printf("B=%d N=%d CTAs=%d active=%d PASS\n",b,n,q.blocks,q.workersPerBlock);
    }
    return 0;
}
