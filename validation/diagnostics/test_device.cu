#include "Diagnostics.H"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
using namespace ugkp::diagnostics;
__global__ void producer(DeviceContext* context) {
    if (!blockMayProceed(context)) return;
    const auto i=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    Record r=makeRecord(Code::NegativeTemperature,701,7,1,EntityKind::Cell,i,-1,static_cast<double>(i));
    recordWarning(context,makeRecord(Code::AccuracyResidual,702));
    recordFatal(context,r);
}
__global__ void consumer(DeviceContext* context, int* touched) {
    if (!blockMayProceed(context)) return;
    __shared__ int values[128]; values[threadIdx.x]=1;
    __syncthreads();
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<4096;i+=gridDim.x*blockDim.x)
        atomicAdd(touched,values[threadIdx.x]);
}
#define CUDA_OK(call) do { const auto e=(call); if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));return 1;} }while(0)
int main() {
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess || count==0) {std::puts("SKIP: no CUDA device; compile-only evidence");return 77;}
    DeviceContext initial{};reset(initial,88,99,4,0.125);
    DeviceContext* context=nullptr;int* touched=nullptr;
    CUDA_OK(cudaMalloc(&context,sizeof(initial)));CUDA_OK(cudaMalloc(&touched,sizeof(int)));
    CUDA_OK(cudaMemcpy(context,&initial,sizeof(initial),cudaMemcpyHostToDevice));CUDA_OK(cudaMemset(touched,0,sizeof(int)));
    cudaStream_t a,b;cudaEvent_t ready;CUDA_OK(cudaStreamCreate(&a));CUDA_OK(cudaStreamCreate(&b));CUDA_OK(cudaEventCreate(&ready));
    cudaGraph_t graph;cudaGraphExec_t executable;
    CUDA_OK(cudaStreamBeginCapture(a,cudaStreamCaptureModeGlobal));
    beginOperationKernel<<<1,1,0,a>>>(context,88,4,0.125);
    producer<<<16,128,0,a>>>(context);consumer<<<16,128,0,a>>>(context,touched);
    CUDA_OK(cudaStreamEndCapture(a,&graph));CUDA_OK(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    CUDA_OK(cudaGraphLaunch(executable,a));CUDA_OK(cudaEventRecord(ready,a));CUDA_OK(cudaStreamWaitEvent(b,ready,0));
    consumer<<<16,128,0,b>>>(context,touched);CUDA_OK(cudaStreamSynchronize(b));
    DeviceContext result{};int touches=-1;CUDA_OK(cudaMemcpy(&result,context,sizeof(result),cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(&touches,touched,sizeof(touches),cudaMemcpyDeviceToHost));
    assert(touches==0 && result.fatalClaim==1 && result.firstFatal.entityIndex==result.firstFatal.observed);
    assert(result.firstFatal.operationSequence==88 && result.firstFatal.step==4);
    assert(result.warnings[1].count>0 && result.warnings[1].count<=2048);
    // Independent operation storage remains healthy and its consumer executes.
    CUDA_OK(cudaMemcpy(context,&initial,sizeof(initial),cudaMemcpyHostToDevice));
    consumer<<<16,128,0,b>>>(context,touched);CUDA_OK(cudaStreamSynchronize(b));
    CUDA_OK(cudaMemcpy(&touches,touched,sizeof(touches),cudaMemcpyDeviceToHost));assert(touches==4096);
    CUDA_OK(cudaGraphExecDestroy(executable));CUDA_OK(cudaGraphDestroy(graph));CUDA_OK(cudaEventDestroy(ready));
    CUDA_OK(cudaStreamDestroy(a));CUDA_OK(cudaStreamDestroy(b));CUDA_OK(cudaFree(touched));CUDA_OK(cudaFree(context));
}
