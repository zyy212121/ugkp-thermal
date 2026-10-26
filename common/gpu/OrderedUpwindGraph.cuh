#ifndef UGKP_GPU_ORDERED_UPWIND_GRAPH_CUH
#define UGKP_GPU_ORDERED_UPWIND_GRAPH_CUH
#include "OrderedUpwindPlan.H"
#include "OrderedUpwindSweep.cuh"
#include "OrderedMaximum.cuh"
#include <climits>
#include <stdexcept>
#include <string>
#include <vector>
namespace ugkp { namespace gpu {
struct UpwindLinearControls {double tolerance=1e-12,relativeTolerance=0,small=1e-15;int maximumIterations=1000;};
struct UpwindLinearReport {
    double initialResidual=0,finalResidual=0,target=0,maximumResidual=0,maximumMagnitude=0,errorValue=0;
    int iterations=0,errorCell=-1,active=0;
    unsigned long long externalFailure=ULLONG_MAX;
};
namespace upwindGraphDetail {
inline void checkedCudaUpwind(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(std::string("CUDA ordered upwind graph: ")+cudaGetErrorString(code));}
template<class T> struct Array {
    T* ptr=nullptr;
    explicit Array(std::size_t n){checkedCudaUpwind(cudaMalloc(reinterpret_cast<void**>(&ptr),std::max(std::size_t(1),n)*sizeof(T)));}
    ~Array(){if(ptr)cudaFree(ptr);}
    Array(const Array&)=delete;Array& operator=(const Array&)=delete;
};
static __global__ void initialize(UpwindLinearReport* report,unsigned int* failure){if(!threadIdx.x){*report={};*failure=UINT_MAX;}}
static __global__ void normalizationFaces(int cells,int faces,const int* owner,const int* neighbour,const double* flux,
    const double* values,double* magnitude,unsigned char* valid){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<cells)valid[i]=1;
    if(i<faces){const double f=flux[i];bool present=false;double value=0;
        if(f>0){value=f*values[owner[i]];present=true;}else if(f<0){value=f*values[neighbour[i]];present=true;}
        magnitude[cells+i]=fabs(value);valid[cells+i]=present;}
}
static __global__ void absoluteResidual(int count,double* values){const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)values[i]=fabs(values[i]);}
static __global__ void maximumResult(OrderedMaximum* maximum,double initial,double* output){if(!threadIdx.x)*output=applyMaximum(initial,*maximum);}
__device__ inline bool needsSweep(const UpwindLinearReport& report,UpwindLinearControls controls){
    return (report.finalResidual>report.target || (report.iterations==0&&
        (controls.maximumIterations<=0||!std::isfinite(report.initialResidual)||std::isnan(report.target))))
        &&report.iterations<(controls.maximumIterations>1?controls.maximumIterations:1);
}
static __global__ void beginSolve(UpwindLinearReport* report,UpwindLinearControls controls,
    const unsigned long long* externalFailure){
    if(threadIdx.x)return;
    report->initialResidual=report->finalResidual=report->maximumResidual/report->maximumMagnitude;
    const double relative=controls.relativeTolerance*report->initialResidual;
    report->target=controls.tolerance>relative?controls.tolerance:relative;
    report->externalFailure=externalFailure?*externalFailure:ULLONG_MAX;
    report->active=report->externalFailure==ULLONG_MAX&&needsSweep(*report,controls);
}
static __global__ void finishSweep(UpwindLinearReport* report,UpwindLinearControls controls,
    const unsigned int* firstFailure,const UpwindIntensityFailure* failures){
    if(threadIdx.x)return;
    if(!report->active)return;
    ++report->iterations;report->finalResidual=report->maximumResidual/report->maximumMagnitude;
    if(*firstFailure!=UINT_MAX){report->errorCell=*firstFailure;report->errorValue=failures[*firstFailure].value;}
    report->active=report->errorCell<0&&needsSweep(*report,controls);
}
// One cooperative block traverses the same dependency levels. Each level is
// a block barrier instead of a separate launch; row work is strided so the
// method has no mesh-size special case. A bounded batch retains host progress.
static __global__ void cooperativeSweeps(OrderedUpwindDevice matrix,int levelCount,const int* levels,
    int faces,const int* owner,const int* neighbour,const double* flux,const double* diagonal,const double* source,
    double* values,double* previous,const double* reference,UpwindLinearControls controls,
    unsigned int* firstFailure,UpwindIntensityFailure* failures,UpwindLinearReport* report){
    __shared__ OrderedMaximum maxima[2][256];
    for(int sweep=0;sweep<8;++sweep){
        if(!report->active)return;
        for(int cell=threadIdx.x;cell<matrix.cells;cell+=blockDim.x)previous[cell]=values[cell];
        __syncthreads();
        for(int level=0;level<levelCount;++level){
            for(int row=levels[level]+threadIdx.x;row<levels[level+1];row+=blockDim.x)
                orderedUpwindCell(matrix.levelCells[row],matrix,diagonal,source,previous,values,0,failures,firstFailure,reference);
            __syncthreads();
        }
        OrderedMaximum residualMax{0,0,0},magnitudeMax{0,0,0};
        const int total=matrix.cells+faces,chunk=(total+255)/256;
        const int begin=threadIdx.x*chunk,end=begin+chunk<total?begin+chunk:total;
        for(int i=begin;i<end;++i){
            if(i<matrix.cells){
                double residual=diagonal[i]*values[i]-source[i];
                for(int j=matrix.offsets[i];j<matrix.offsets[i+1];++j){const int edge=matrix.residualOrder[j];
                    if(matrix.subtract[edge])residual+=matrix.flux[edge]*values[matrix.neighbour[edge]];
                    else residual-=matrix.flux[edge]*values[matrix.neighbour[edge]];}
                residualMax=joinMaximum(residualMax,maximumSegment(fabs(residual)));
                const double a=fabs(diagonal[i]*values[i]),b=fabs(source[i]);
                magnitudeMax=joinMaximum(magnitudeMax,maximumSegment(a>b?a:b));
            }else{const int face=i-matrix.cells;const double f=flux[face];
                if(f>0)magnitudeMax=joinMaximum(magnitudeMax,maximumSegment(fabs(f*values[owner[face]])));
                else if(f<0)magnitudeMax=joinMaximum(magnitudeMax,maximumSegment(fabs(f*values[neighbour[face]])));
            }
        }
        maxima[0][threadIdx.x]=residualMax;maxima[1][threadIdx.x]=magnitudeMax;__syncthreads();
        for(int stride=1;stride<256;stride*=2){if(threadIdx.x%(2*stride)==0){
            maxima[0][threadIdx.x]=joinMaximum(maxima[0][threadIdx.x],maxima[0][threadIdx.x+stride]);
            maxima[1][threadIdx.x]=joinMaximum(maxima[1][threadIdx.x],maxima[1][threadIdx.x+stride]);}
            __syncthreads();}
        if(!threadIdx.x){
            report->maximumResidual=applyMaximum(0,maxima[0][0]);report->maximumMagnitude=applyMaximum(controls.small,maxima[1][0]);
            ++report->iterations;report->finalResidual=report->maximumResidual/report->maximumMagnitude;
            if(*firstFailure!=UINT_MAX){report->errorCell=*firstFailure;report->errorValue=failures[*firstFailure].value;}
            report->active=report->errorCell<0&&needsSweep(*report,controls);
        }
        __syncthreads();
    }
}

}
// Immutable launch graph for the original sequential-cell linear iteration.
// All pointers are device pointers and remain owned by the caller, except the
// compact report and scratch storage owned here. Launches use the caller stream.
class OrderedUpwindGraph {
    using Report=UpwindLinearReport;
    int cells_,faces_;bool cooperative_;
    upwindGraphDetail::Array<double> residual_,magnitude_;
    upwindGraphDetail::Array<unsigned char> valid_;
    upwindGraphDetail::Array<OrderedMaximum> partialA_,partialB_;
    upwindGraphDetail::Array<unsigned int> failure_;
    upwindGraphDetail::Array<int> levels_;
    upwindGraphDetail::Array<UpwindIntensityFailure> failures_;
    upwindGraphDetail::Array<Report> report_;
    cudaGraph_t graph_=nullptr,initializeGraph_=nullptr;
    cudaGraphExec_t executable_=nullptr,initializeExecutable_=nullptr;
    Report* hostReport_=nullptr;
    void maximum(int count,double* values,unsigned char* valid,double initial,double* output,cudaStream_t stream){
        int blocks=std::max(1,(count+255)/256);
        orderedMaximumPartials<<<blocks,256,0,stream>>>(count,values,valid,partialA_.ptr);
        auto* current=partialA_.ptr;auto* next=partialB_.ptr;
        while(blocks>1){const int n=(blocks+255)/256;orderedMaximumMerge<<<n,256,0,stream>>>(blocks,current,next);std::swap(current,next);blocks=n;}
        upwindGraphDetail::maximumResult<<<1,1,0,stream>>>(current,initial,output);
    }
    void residual(OrderedUpwindDevice matrix,const int* owner,const int* neighbour,const double* flux,
        const double* diagonal,const double* source,const double* values,UpwindLinearControls controls,cudaStream_t stream){
        upwindResidual<<<std::max(1,(cells_+127)/128),128,0,stream>>>(matrix,diagonal,source,values,residual_.ptr,magnitude_.ptr);
        upwindGraphDetail::normalizationFaces<<<std::max(1,(std::max(cells_,faces_)+127)/128),128,0,stream>>>(cells_,faces_,owner,neighbour,flux,values,magnitude_.ptr,valid_.ptr);
        upwindGraphDetail::absoluteResidual<<<std::max(1,(cells_+127)/128),128,0,stream>>>(cells_,residual_.ptr);
        maximum(cells_,residual_.ptr,nullptr,0,&report_.ptr->maximumResidual,stream);
        maximum(cells_+faces_,magnitude_.ptr,valid_.ptr,controls.small,&report_.ptr->maximumMagnitude,stream);
    }
public:
    OrderedUpwindGraph(const OrderedUpwindPlan& plan,OrderedUpwindDevice matrix,int faces,
        const int* owner,const int* neighbour,const double* flux,const double* diagonal,const double* source,
        double* values,double* previous,const double* reference,UpwindLinearControls controls,
        const unsigned long long* externalFailure=nullptr,bool concurrentDirections=false):cells_(plan.cells),faces_(faces),cooperative_(concurrentDirections),
        residual_(cells_),magnitude_(cells_+faces_),valid_(cells_+faces_),
        partialA_(std::max(1,(cells_+faces_+255)/256)),partialB_(std::max(1,(cells_+faces_+255)/256)),
        failure_(1),levels_(plan.levelOffsets.size()),failures_(cells_),report_(1){
        using upwindGraphDetail::checkedCudaUpwind;cudaStream_t stream=nullptr;
        checkedCudaUpwind(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        try {
            checkedCudaUpwind(cudaMallocHost(reinterpret_cast<void**>(&hostReport_),sizeof(Report)));
            checkedCudaUpwind(cudaMemcpyAsync(levels_.ptr,plan.levelOffsets.data(),plan.levelOffsets.size()*sizeof(int),cudaMemcpyHostToDevice,stream));
            checkedCudaUpwind(cudaStreamSynchronize(stream));
            checkedCudaUpwind(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            upwindGraphDetail::initialize<<<1,1,0,stream>>>(report_.ptr,failure_.ptr);
            residual(matrix,owner,neighbour,flux,diagonal,source,values,controls,stream);
            upwindGraphDetail::beginSolve<<<1,1,0,stream>>>(report_.ptr,controls,externalFailure);
            checkedCudaUpwind(cudaStreamEndCapture(stream,&initializeGraph_));
            checkedCudaUpwind(cudaGraphInstantiateWithFlags(&initializeExecutable_,initializeGraph_,0));
            checkedCudaUpwind(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            // Independent angular solves can occupy different SMs. Ordered
            // angular Gauss-Seidel instead spreads each level across the GPU.
            if(cooperative_){
                upwindGraphDetail::cooperativeSweeps<<<1,256,0,stream>>>(matrix,plan.levelOffsets.size()-1,levels_.ptr,
                    faces_,owner,neighbour,flux,diagonal,source,values,previous,reference,controls,failure_.ptr,failures_.ptr,report_.ptr);
            }else{
                snapshotUpwind<<<std::max(1,(cells_+127)/128),128,0,stream>>>(cells_,values,previous,&report_.ptr->active);
                for(std::size_t level=0;level+1<plan.levelOffsets.size();++level){const int begin=plan.levelOffsets[level],end=plan.levelOffsets[level+1];
                    orderedUpwindLevel<<<(end-begin+127)/128,128,0,stream>>>(begin,end,matrix,diagonal,source,previous,values,0,failures_.ptr,failure_.ptr,reference,&report_.ptr->active);}
                residual(matrix,owner,neighbour,flux,diagonal,source,values,controls,stream);
                upwindGraphDetail::finishSweep<<<1,1,0,stream>>>(report_.ptr,controls,failure_.ptr,failures_.ptr);
            }
            checkedCudaUpwind(cudaStreamEndCapture(stream,&graph_));checkedCudaUpwind(cudaGraphInstantiateWithFlags(&executable_,graph_,0));
            checkedCudaUpwind(cudaStreamDestroy(stream));

        } catch(...) {
            cudaStreamCaptureStatus status;
            if(cudaStreamIsCapturing(stream,&status)==cudaSuccess&&status!=cudaStreamCaptureStatusNone){cudaGraph_t abandoned=nullptr;cudaStreamEndCapture(stream,&abandoned);if(abandoned)cudaGraphDestroy(abandoned);}
            release();cudaStreamDestroy(stream);throw;
        }
    }
    ~OrderedUpwindGraph(){release();}
    OrderedUpwindGraph(const OrderedUpwindGraph&)=delete;OrderedUpwindGraph& operator=(const OrderedUpwindGraph&)=delete;
    // Batches are a scheduling choice only: the device active flag freezes
    // the first converged/failed iterate, including within a queued batch.
    void begin(cudaStream_t stream){upwindGraphDetail::checkedCudaUpwind(cudaGraphLaunch(initializeExecutable_,stream));}
    void advance(cudaStream_t stream,int count=-1){if(count<0)count=cooperative_?1:8;for(int i=0;i<count;++i)upwindGraphDetail::checkedCudaUpwind(cudaGraphLaunch(executable_,stream));}
    const Report& collect(cudaStream_t stream){
        upwindGraphDetail::checkedCudaUpwind(cudaMemcpyAsync(hostReport_,report_.ptr,sizeof(Report),cudaMemcpyDeviceToHost,stream));
        upwindGraphDetail::checkedCudaUpwind(cudaStreamSynchronize(stream));return *hostReport_;
    }
    const Report& launch(cudaStream_t stream){
        begin(stream);advance(stream,1);
        while(collect(stream).active)advance(stream);
        return *hostReport_;
    }
private:
    void release()noexcept{
        if(executable_)cudaGraphExecDestroy(executable_);if(initializeExecutable_)cudaGraphExecDestroy(initializeExecutable_);
        if(graph_)cudaGraphDestroy(graph_);if(initializeGraph_)cudaGraphDestroy(initializeGraph_);if(hostReport_)cudaFreeHost(hostReport_);
    }
public:
    const Report* report()const noexcept{return report_.ptr;}
};
}}
#endif
