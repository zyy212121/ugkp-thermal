#include "gpu/SparseConjugateGradient.H"
#include "gpu/ExecutionGraph.cuh"
#include <cuda_runtime.h>
#if __has_include(<cuda_profiler_api.h>)
#include <cuda_profiler_api.h>
#else
extern "C" cudaError_t CUDARTAPI cudaProfilerStart(void);
extern "C" cudaError_t CUDARTAPI cudaProfilerStop(void);
#endif
#include <cmath>
#include <cstdio>
#include <stdexcept>
using namespace ugkp::gpu;
static void check(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
// The exact solution of [3 -1; -1 3] x = scale*[2 4] is
// scale*[5/4 7/4]. Source updates and result consumption stay on device.
__global__ void source(double* rhs,int iteration){if(!threadIdx.x){rhs[0]=2*(iteration+1);rhs[1]=4*(iteration+1);}}
__global__ void consume(const double* x,const ConjugateGradientReport* report,int iteration,int* failure){
    if(threadIdx.x)return;
    if(!report->converged||report->singular||report->iterations<1||report->iterations>2
       ||fabs(x[0]-1.25*(iteration+1))>1e-11||fabs(x[1]-1.75*(iteration+1))>1e-11)
        *failure=iteration+1;
}
__global__ void exactInitialSource(const double* x,double* rhs){if(!threadIdx.x){rhs[0]=3*x[0]-x[1];rhs[1]=3*x[1]-x[0];}}
__global__ void consumeInitiallySolved(const ConjugateGradientReport* report,int* failure){
    if(!threadIdx.x&&(!report->converged||report->singular||report->iterations!=0||report->initialResidual!=0))*failure=100;
}
__global__ void nestedSource(double* rhs,const int* iteration){if(!threadIdx.x){rhs[0]=2*(*iteration+1);rhs[1]=4*(*iteration+1);}}
__global__ void nestedConsume(const double* x,const ConjugateGradientReport* report,int* iteration,int* failure,cudaGraphConditionalHandle handle){
    if(threadIdx.x)return;const int i=*iteration;
    if(!report->converged||report->singular||report->iterations<1||report->iterations>2
       ||fabs(x[0]-1.25*(i+1))>1e-11||fabs(x[1]-1.75*(i+1))>1e-11)*failure=200+i;
    ++*iteration;cudaGraphSetConditional(handle,*iteration<8);
}
int main(){
    double *diagonal,*off,*rhs,*x;int* failure;
    check(cudaMalloc(&diagonal,2*sizeof(double)));check(cudaMalloc(&off,sizeof(double)));
    check(cudaMalloc(&rhs,2*sizeof(double)));check(cudaMalloc(&x,2*sizeof(double)));check(cudaMalloc(&failure,sizeof(int)));
    const double hostDiagonal[]={3,3},hostOff=-1;
    check(cudaMemcpy(diagonal,hostDiagonal,sizeof(hostDiagonal),cudaMemcpyHostToDevice));
    check(cudaMemcpy(off,&hostOff,sizeof(hostOff),cudaMemcpyHostToDevice));
    check(cudaMemset(x,0,2*sizeof(double)));check(cudaMemset(failure,0,sizeof(int)));
    const int owner[]={0},neighbour[]={1};SparseLduView topology;topology.rows=2;topology.faces=1;
    topology.owner=owner;topology.neighbour=neighbour;
    SparseDeviceLduValues values{diagonal,off,off,rhs,nullptr,nullptr,nullptr};
    ConjugateGradientControls controls;controls.tolerance=1e-12;controls.maximumIterations=20;
    SparseConjugateGradient solver;solver.configureResidentLocalLdu(topology,values,controls);
    int* outerIteration;check(cudaMalloc(&outerIteration,sizeof(int)));
    cudaStream_t outerStream;check(cudaStreamCreateWithFlags(&outerStream,cudaStreamNonBlocking));
    cudaGraph_t outer,body;cudaGraphExec_t executable;cudaGraphConditionalHandle handle;
    check(cudaGraphCreate(&outer,0));check(cudaGraphConditionalHandleCreate(&handle,outer,1,cudaGraphCondAssignDefault));
    using executionGraphDetail::appendCaptured;using executionGraphDetail::addWhile;
    const auto initial=appendCaptured(outer,{},outerStream,[&]{check(cudaMemsetAsync(outerIteration,0,sizeof(int),outerStream));});
    addWhile(outer,handle,{initial},body);
    const auto prepared=appendCaptured(body,{},outerStream,[&]{nestedSource<<<1,1,0,outerStream>>>(rhs,outerIteration);});
    const auto solved=solver.appendResidentLocalLdu(body,prepared,x);
    appendCaptured(body,{solved},outerStream,[&]{nestedConsume<<<1,1,0,outerStream>>>(x,solver.deviceReport(),outerIteration,failure,handle);});
    check(cudaGraphInstantiateWithFlags(&executable,outer,0));
    cudaEvent_t outerReady;check(cudaEventCreateWithFlags(&outerReady,cudaEventDisableTiming));
    check(cudaDeviceSynchronize());check(cudaProfilerStart());
    for(int iteration=0;iteration<8;++iteration){
        source<<<1,1>>>(rhs,iteration);
        const auto view=solver.enqueueResidentLocalLdu(x);
        // enqueue orders publication back to values.readyStream (default here).
        consume<<<1,1>>>(x,view.report,iteration,failure);
    }
    check(cudaEventRecord(outerReady,nullptr));check(cudaStreamWaitEvent(outerStream,outerReady,0));
    check(cudaGraphLaunch(executable,outerStream));
    check(cudaEventRecord(outerReady,outerStream));check(cudaStreamWaitEvent(nullptr,outerReady,0));
    exactInitialSource<<<1,1>>>(x,rhs);
    const auto last=solver.enqueueResidentLocalLdu(x);
    consumeInitiallySolved<<<1,1>>>(last.report,failure);
    check(cudaProfilerStop());
    // Explicit terminal publication is outside the profiled advance interval.
    check(cudaDeviceSynchronize());int failed=-1;double result[2];
    check(cudaMemcpy(&failed,failure,sizeof(failed),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(result,x,sizeof(result),cudaMemcpyDeviceToHost));
    const auto report=solver.collectResidentLocalReport();
    if(failed||report.iterations!=0||std::abs(result[0]-10)>1e-11||std::abs(result[1]-14)>1e-11)return 1;
    std::puts("PASS: 17 resident solves including nested device-controlled PCG; explicit terminal publication");
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(outer));check(cudaStreamDestroy(outerStream));check(cudaEventDestroy(outerReady));check(cudaFree(outerIteration));
    check(cudaFree(diagonal));check(cudaFree(off));check(cudaFree(rhs));check(cudaFree(x));check(cudaFree(failure));
}
