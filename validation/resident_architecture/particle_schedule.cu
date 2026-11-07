#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "../../common/GpuCsrConditionalGraph.H"
#include "../../common/GpuParticleExecutionContext.H"
#define UGKP_RESIDENT_CSR_CHOICE 1
#define GPU_DIRECTORY_PARAMETER_TYPE int
#define GPU_DIRECTORY_SELECTOR directory
#define GPU_DIRECTORY_ARGUMENT directory
#define GPU_DIRECTORY_HAS_BASE_ONLY 0
#define GPU_DIRECTORY_FULL 0
#define GPU_AUTO_THRESHOLD_FIELD threshold
struct DeviceState {
    DeviceState* deviceState=nullptr;
    ugkp::gpu::CsrAutomaticRuntime* csrAutomaticRuntime=nullptr;
    cudaStream_t particleCaptureStream=nullptr;cudaGraph_t particleCaptureRoot=nullptr;
    int csrHeavyReductionMode=2,particleCapacity=1024,csrHeavyAutoInterval=3,nCells=1;
    int csrHeavyReductionEnabled=1,csrHeavyReductionActive=1,csrTasksReady=0,csrPreparedDirectoryKind=0;
    int *csrMaximumOccupancy=nullptr,*cellParticleOffset=nullptr,*preBaseCellOffset=nullptr;
    int threshold=10,nextThreshold=10;
};
static void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);}}
static void setLastError(const char* label,cudaError_t e){std::fprintf(stderr,"%s: %s\n",label,cudaGetErrorString(e));}
__global__ void configure(DeviceState* s,bool scheduled){if(!scheduled||s->csrAutomaticRuntime->inspect)s->threshold=s->nextThreshold;}
static int policy(DeviceState* s,int,bool scheduled){configure<<<1,1,0,s->particleCaptureStream>>>(s->deviceState,scheduled);return int(cudaGetLastError()!=cudaSuccess);}
#define GPU_AUTO_UPDATE_POLICY(s, kind) policy(s,kind,false)
#define GPU_AUTO_INSPECTION_UPDATE_POLICY(s, kind) policy(s,kind,true)
#include "../../common/GpuAutomaticCsrSchedule.cuh"
int prepareCsrSegmentedReductionTasks(DeviceState* s,int,int directory){s->csrTasksReady=1;s->csrPreparedDirectoryKind=directory;return 0;}
int prepareCsrSegmentedReductionTasksOnInspection(DeviceState* s,int block,int directory){return prepareCsrSegmentedReductionTasks(s,block,directory);}
__global__ void inputs(DeviceState* s,int count,int threshold){s->cellParticleOffset[0]=0;s->cellParticleOffset[1]=count;s->nextThreshold=threshold;}
int main(){
    int scenarios=0;
    for(int interval:{1,3,7})for(int captured:{0,1}){
        DeviceState s;s.csrHeavyAutoInterval=interval;
        check(cudaMalloc(&s.deviceState,sizeof(s)));check(cudaMalloc(&s.csrAutomaticRuntime,sizeof(*s.csrAutomaticRuntime)));
        check(cudaMalloc(&s.csrMaximumOccupancy,sizeof(int)));check(cudaMalloc(&s.cellParticleOffset,2*sizeof(int)));
        ugkp::gpu::CsrAutomaticRuntime runtime{};check(cudaMemcpy(s.csrAutomaticRuntime,&runtime,sizeof(runtime),cudaMemcpyHostToDevice));
        check(cudaMemcpy(s.deviceState,&s,sizeof(s),cudaMemcpyHostToDevice));
        cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
        if(captured){s.particleCaptureStream=stream;check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            if(runAutomaticCsrSchedule(&s,64,0))return 2;
            check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiateWithFlags(&exec,graph,0));s.particleCaptureStream=nullptr;}
        int active=1,threshold=10;
        for(int step=1;step<=257;++step){
            const int count=(step*13)%31,next=(step*7)%19+1;const bool inspect=step==1||step%interval==0;
            if(inspect){threshold=next;active=count>threshold;}
            inputs<<<1,1>>>(s.deviceState,count,next);
            if(captured)check(cudaGraphLaunch(exec,nullptr));else if(runAutomaticCsrSchedule(&s,64,0))return 3;
            DeviceState actual;check(cudaMemcpy(&actual,s.deviceState,sizeof(actual),cudaMemcpyDeviceToHost));
            check(cudaMemcpy(&runtime,s.csrAutomaticRuntime,sizeof(runtime),cudaMemcpyDeviceToHost));
            if(runtime.advanceCount!=static_cast<unsigned long long>(step)||runtime.inspect!=inspect||runtime.active!=active||actual.threshold!=threshold){
                std::fprintf(stderr,"schedule mismatch: interval %d captured %d step %d\n",interval,captured,step);return 4;}
        }
        if(exec)check(cudaGraphExecDestroy(exec));if(graph)check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
        check(cudaFree(s.deviceState));check(cudaFree(s.csrAutomaticRuntime));check(cudaFree(s.csrMaximumOccupancy));check(cudaFree(s.cellParticleOffset));++scenarios;
    }
    std::printf("PASS %d schedules, 1542 decisions, thresholds change only at the original inspection cadence\n",scenarios);
}
