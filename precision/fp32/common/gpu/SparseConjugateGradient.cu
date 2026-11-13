#include "SparseConjugateGradient.H"
#include "ExecutionGraph.cuh"
#include <cuda_runtime.h>
#include "PinnedTransferQueue.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
namespace ugkp { namespace gpu { namespace {
void checked(cudaError_t code) {
    if(code!=cudaSuccess)throw std::runtime_error(std::string("CUDA sparse solve: ")+cudaGetErrorString(code));
}
template<class T> struct DeviceArray {
    T* ptr=nullptr;std::size_t capacity=0;
    ~DeviceArray(){if(ptr)cudaFree(ptr);}
    void reserve(std::size_t count){
        if(count<=capacity)return;
        T* replacement=nullptr;checked(cudaMalloc(reinterpret_cast<void**>(&replacement),count*sizeof(T)));
        if(ptr)cudaFree(ptr);ptr=replacement;capacity=count;
    }
    template<class Host> void upload(const Host* host,std::size_t count,PinnedTransferQueue& queue){
        reserve(count);queue.uploadConverted(ptr,host,count);
    }
};
constexpr int threads=256;
__global__ void multiply(int n,const int* offsets,const int* columns,const GpuReal* values,const GpuReal* x,GpuReal* y,const GpuReal* stop){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||(stop&&stop[7]))return;
    GpuReal value=values[offsets[i]]*x[i];
    for(int e=offsets[i]+1;e<offsets[i+1];++e)value+=values[e]*x[columns[e]];
    y[i]=value;
}
__global__ void assembleLdu(int n,const int* offsets,const int* entrySource,const int* entryKind,
    const int* boundaryOffsets,const int* boundaryIndices,const int* boundaryNeighbour,
    const GpuReal* diagonal,const GpuReal* upper,const GpuReal* lower,const GpuReal* source,
    const GpuReal* boundaryDiagonal,const GpuReal* boundarySource,GpuReal* values,GpuReal* rhs,GpuReal* rowSum,
    const GpuReal* boundaryJump=nullptr,GpuReal* affine=nullptr){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    GpuReal d=diagonal[i],b=source[i],jump=0;
    for(int j=boundaryOffsets[i];j<boundaryOffsets[i+1];++j){
        const int f=boundaryIndices[j];d+=boundaryDiagonal[f];
        if(boundaryNeighbour[f]==-1)b+=boundarySource[f];
        if(boundaryJump)jump+=boundarySource[f]*boundaryJump[f];
    }
    values[offsets[i]]=d;rhs[i]=b;GpuReal sum=d;
    for(int j=offsets[i]+1;j<offsets[i+1];++j){
        const int f=entrySource[j],kind=entryKind[j];
        const GpuReal value=kind==1?lower[f]:kind==2?upper[f]:-boundarySource[f];
        values[j]=value;if(kind!=3)sum+=value;
    }
    for(int j=boundaryOffsets[i];j<boundaryOffsets[i+1];++j){
        const int f=boundaryIndices[j];if(boundaryNeighbour[f]!=-1)sum-=boundarySource[f];
    }
    rowSum[i]=sum;if(affine)affine[i]=jump;
}
// Preserve native affine initial-product and normFactor semantics. Subsequent
// Krylov matvecs remain linear; moving jump into b alone would change the
// OpenFOAM residual normalization and relative stopping criterion.
__global__ void addInitialAffine(int n,const GpuReal* affine,GpuReal* product){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)product[i]+=affine[i];
}
__global__ void initialResidual(int n,const GpuReal* source,const GpuReal* ax,GpuReal* r){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)r[i]=source[i]-ax[i];
}
__global__ void precondition(int n,const int* offsets,const GpuReal* values,const GpuReal* r,GpuReal* z,const GpuReal* stop){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&!(stop&&stop[7]))z[i]=r[i]/values[offsets[i]];
}
// Recurrence scalars: rho, pAp, previous rho, alpha, |r|_1, singular, completed iterations.
__global__ void direction(int n,const GpuReal* z,const GpuReal* recurrence,GpuReal* p){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n&&!recurrence[7])p[i]=recurrence[6]==0?z[i]:z[i]+(recurrence[0]/recurrence[2])*p[i];
}
struct LocalSolveParameters {GpuReal norm,initial,mean;ConjugateGradientControls controls;};
__global__ void initializeLocalParameters(int count,ConjugateGradientControls controls,
    LocalSolveParameters* parameters,GpuReal* recurrence,const ConjugateGradientControls* deviceControls){
    if(blockIdx.x||threadIdx.x)return;
    *parameters={0,0,0,deviceControls?*deviceControls:controls};
    for(int i=0;i<8;++i)recurrence[i]=0;
}
__global__ void normalizeLocalMean(int count,GpuReal* mean){
    if(!blockIdx.x&&!threadIdx.x)*mean/=count?count:1;
}
__global__ void initializeLocalCondition(LocalSolveParameters* parameters,GpuReal* recurrence,
    cudaGraphConditionalHandle handle){
    if(blockIdx.x||threadIdx.x)return;
    parameters->norm+=parameters->controls.normalizationSmall;
    parameters->initial=recurrence[4]/parameters->norm;
    const auto& c=parameters->controls;
    const bool converged=parameters->initial<c.tolerance ||
        (c.relativeTolerance>c.normalizationSmall&&parameters->initial<c.relativeTolerance*parameters->initial);
    recurrence[7]=c.minimumIterations<=0&&converged;
    cudaGraphSetConditional(handle,recurrence[7]?0:1);
}
__global__ void publishLocalReport(const LocalSolveParameters* parameters,const GpuReal* recurrence,
    ConjugateGradientReport* report){
    if(blockIdx.x||threadIdx.x)return;
    const auto& c=parameters->controls;const GpuReal residual=recurrence[4]/parameters->norm;
    report->iterations=int(recurrence[6]);report->initialResidual=parameters->initial;report->finalResidual=residual;
    report->singular=recurrence[5]!=0;
    report->converged=residual<c.tolerance ||
        (c.relativeTolerance>c.normalizationSmall&&residual<c.relativeTolerance*parameters->initial);
}
__global__ void prepareStep(GpuReal* recurrence,GpuReal norm,GpuReal singularSmall,const LocalSolveParameters* parameters=nullptr){
    if(blockIdx.x||threadIdx.x||recurrence[7])return;
    if(parameters){norm=parameters->norm;singularSmall=parameters->controls.singularSmall;}
    recurrence[5]=fabsf(recurrence[1])/norm<singularSmall;
    if(!recurrence[5]){recurrence[3]=recurrence[0]/recurrence[1];recurrence[6]+=1;}
    recurrence[2]=recurrence[0];
}
__global__ void update(int n,const GpuReal* recurrence,const GpuReal* p,const GpuReal* ap,GpuReal* x,GpuReal* r){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n&&!recurrence[5]&&!recurrence[7]){x[i]+=recurrence[3]*p[i];r[i]-=recurrence[3]*ap[i];}
}
// Fixed reduction tree; no floating-point atomics or scheduling-dependent sums.
// mode: sum, absolute sum, dot, finite-volume residual normalization.
__global__ void partialReduction(int n,int mode,const GpuReal* a,const GpuReal* b,
    const GpuReal* rowSum,GpuReal mean,GpuReal* partial,const GpuReal* stop,const GpuReal* residentMean=nullptr){
    if(residentMean)mean=*residentMean;
    if(stop&&stop[7])return;
    __shared__ GpuReal values[threads];GpuReal v=0;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
        if(mode==0)v+=a[i];
        else if(mode==1)v+=fabsf(a[i]);
        else if(mode==2)v+=a[i]*b[i];
        else {const GpuReal reference=rowSum[i]*mean;v+=fabsf(a[i]-reference)+fabsf(b[i]-reference);}
    }
    values[threadIdx.x]=v;__syncthreads();
    for(int stride=threads/2;stride;stride/=2){if(threadIdx.x<stride)values[threadIdx.x]+=values[threadIdx.x+stride];__syncthreads();}
    if(threadIdx.x==0)partial[blockIdx.x]=values[0];
}
__global__ void finishReduction(int n,const GpuReal* partial,GpuReal* result,const GpuReal* stop){
    if(stop&&stop[7])return;
    __shared__ GpuReal values[threads];GpuReal v=0;
    for(int i=threadIdx.x;i<n;i+=threads)v+=partial[i];
    values[threadIdx.x]=v;__syncthreads();
    for(int stride=threads/2;stride;stride/=2){if(threadIdx.x<stride)values[threadIdx.x]+=values[threadIdx.x+stride];__syncthreads();}
    if(threadIdx.x==0)*result=values[0];
}
__global__ void gatherInterface(int count,const int* cells,const GpuReal* x,const GpuReal* y,GpuReal* packed){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){packed[i]=x[cells[i]];packed[count+i]=y[cells[i]];}
}
__global__ void scatterInterface(int count,const int* cells,const GpuReal* packed,GpuReal* y){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)y[cells[i]]=packed[i];
}
// Keep each reduction's original grid-stride order and binary FP32 tree, while
// producing its terms in the vector/matrix kernel that already owns the row.
__device__ GpuReal localBlockSum(GpuReal value,GpuReal* scratch){
    scratch[threadIdx.x]=value;__syncthreads();
    for(int stride=threads/2;stride;stride/=2){
        if(threadIdx.x<stride)scratch[threadIdx.x]+=scratch[threadIdx.x+stride];
        __syncthreads();
    }
    return scratch[0];
}
__global__ void preconditionAndDot(int n,const int* offsets,const GpuReal* values,
    const GpuReal* r,GpuReal* z,GpuReal* partial,const GpuReal* recurrence){
    if(recurrence[7])return;
    __shared__ GpuReal scratch[threads];GpuReal sum=0;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
        const GpuReal value=r[i]/values[offsets[i]];z[i]=value;sum+=value*r[i];
    }
    sum=localBlockSum(sum,scratch);if(!threadIdx.x)partial[blockIdx.x]=sum;
}
__global__ void multiplyAndDot(int n,const int* offsets,const int* columns,const GpuReal* values,
    const GpuReal* p,GpuReal* ap,GpuReal* partial,const GpuReal* recurrence){
    if(recurrence[7])return;
    __shared__ GpuReal scratch[threads];GpuReal sum=0;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
        GpuReal value=values[offsets[i]]*p[i];
        for(int e=offsets[i]+1;e<offsets[i+1];++e)value+=values[e]*p[columns[e]];
        ap[i]=value;sum+=value*p[i];
    }
    sum=localBlockSum(sum,scratch);if(!threadIdx.x)partial[blockIdx.x]=sum;
}
__global__ void updatePreconditionAndNorm(int n,const int* offsets,const GpuReal* values,
    const GpuReal* recurrence,const GpuReal* p,const GpuReal* ap,GpuReal* x,GpuReal* r,GpuReal* z,GpuReal* partial){
    if(recurrence[7])return;
    __shared__ GpuReal scratch[threads];GpuReal norm=0,rho=0;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
        GpuReal value=r[i];
        if(!recurrence[5]){x[i]+=recurrence[3]*p[i];value-=recurrence[3]*ap[i];r[i]=value;}
        const GpuReal preconditioned=value/values[offsets[i]];z[i]=preconditioned;
        norm+=fabsf(value);rho+=preconditioned*value;
    }
    norm=localBlockSum(norm,scratch);if(!threadIdx.x)partial[blockIdx.x]=norm;
    __syncthreads();
    rho=localBlockSum(rho,scratch);if(!threadIdx.x)partial[gridDim.x+blockIdx.x]=rho;
}

template<int Stage> __global__ void finishLocalReduction(int n,const GpuReal* partial,
    GpuReal* recurrence,const LocalSolveParameters* parameters,cudaGraphConditionalHandle handle){
    if(recurrence[7])return;
    __shared__ GpuReal scratch[threads];GpuReal sum=0;
    for(int i=threadIdx.x;i<n;i+=threads)sum+=partial[i];
    sum=localBlockSum(sum,scratch);
    GpuReal nextRho=0;
    if constexpr(Stage==2){
        for(int i=threadIdx.x;i<n;i+=threads)nextRho+=partial[n+i];
        __syncthreads();
        nextRho=localBlockSum(nextRho,scratch);
    }
    if(threadIdx.x)return;
    if constexpr(Stage==0)recurrence[0]=sum;
    else if constexpr(Stage==1){
        recurrence[1]=sum;
        recurrence[5]=fabsf(sum)/parameters->norm<parameters->controls.singularSmall;
        if(!recurrence[5]){recurrence[3]=recurrence[0]/sum;recurrence[6]+=1;}
        recurrence[2]=recurrence[0];
    }else{
        recurrence[4]=sum;recurrence[0]=nextRho;
        const auto& controls=parameters->controls;
        const GpuReal residual=sum/parameters->norm;
        const bool converged=residual<controls.tolerance ||
            (controls.relativeTolerance>controls.normalizationSmall&&residual<controls.relativeTolerance*parameters->initial);
        recurrence[7]=recurrence[5] ||
            ((recurrence[6]>=controls.maximumIterations||converged)&&recurrence[6]>=controls.minimumIterations);
        cudaGraphSetConditional(handle,recurrence[7]?0:1);
    }
}

// A matrix with at most one row per lane needs no inter-block dependencies.
// Keep the identical CSR order and 256-lane reduction tree, and synchronize
// its complete PCG recurrence inside the owning block instead of dispatching
// a graph node for each vector operation. No mesh/physics assumption is used.
__global__ void solveLocalBlock(int n,const int* offsets,const int* columns,const GpuReal* values,
    const GpuReal* rowSum,const GpuReal* b,const GpuReal* affine,GpuReal* x,GpuReal* r,GpuReal* z,GpuReal* p,GpuReal* ap,
    ConjugateGradientControls settings,const ConjugateGradientControls* deviceSettings,
    LocalSolveParameters* parameters,GpuReal* recurrence,ConjugateGradientReport* report){
    __shared__ GpuReal scratch[threads];const int i=threadIdx.x;
    if(!i){*parameters={0,0,0,deviceSettings?*deviceSettings:settings};for(int j=0;j<8;++j)recurrence[j]=0;}
    __syncthreads();
    GpuReal sum=localBlockSum(i<n?x[i]:0.f,scratch);
    if(!i)parameters->mean=sum/(n?n:1);
    __syncthreads();
    GpuReal product=0,normTerm=0,residual=0;
    if(i<n){
        product=values[offsets[i]]*x[i];
        for(int e=offsets[i]+1;e<offsets[i+1];++e)product+=values[e]*x[columns[e]];
        if(affine)product+=affine[i];
        ap[i]=product;residual=b[i]-product;r[i]=residual;
        const GpuReal reference=rowSum[i]*parameters->mean;
        normTerm=fabsf(product-reference)+fabsf(b[i]-reference);
    }
    sum=localBlockSum(normTerm,scratch);
    if(!i)parameters->norm=sum+parameters->controls.normalizationSmall;
    __syncthreads();
    sum=localBlockSum(fabsf(residual),scratch);
    if(!i){
        recurrence[4]=sum;parameters->initial=sum/parameters->norm;
        const auto& c=parameters->controls;
        const bool converged=parameters->initial<c.tolerance ||
            (c.relativeTolerance>c.normalizationSmall&&parameters->initial<c.relativeTolerance*parameters->initial);
        recurrence[7]=c.minimumIterations<=0&&converged;
    }
    __syncthreads();
    while(!recurrence[7]){
        GpuReal preconditioned=0;if(i<n){preconditioned=r[i]/values[offsets[i]];z[i]=preconditioned;}
        sum=localBlockSum(i<n?preconditioned*r[i]:0.f,scratch);
        if(!i)recurrence[0]=sum;
        __syncthreads();
        if(i<n)p[i]=recurrence[6]==0?preconditioned:preconditioned+(recurrence[0]/recurrence[2])*p[i];
        __syncthreads();
        product=0;
        if(i<n){
            product=values[offsets[i]]*p[i];
            for(int e=offsets[i]+1;e<offsets[i+1];++e)product+=values[e]*p[columns[e]];
            ap[i]=product;
        }
        sum=localBlockSum(i<n?product*p[i]:0.f,scratch);
        if(!i){
            recurrence[1]=sum;recurrence[5]=fabsf(sum)/parameters->norm<parameters->controls.singularSmall;
            if(!recurrence[5]){recurrence[3]=recurrence[0]/sum;recurrence[6]+=1;}
            recurrence[2]=recurrence[0];
        }
        __syncthreads();
        if(i<n&&!recurrence[5]){x[i]+=recurrence[3]*p[i];r[i]-=recurrence[3]*ap[i];}
        sum=localBlockSum(i<n?fabsf(r[i]):0.f,scratch);
        if(!i){
            recurrence[4]=sum;const auto& c=parameters->controls;const GpuReal value=sum/parameters->norm;
            const bool converged=value<c.tolerance||(c.relativeTolerance>c.normalizationSmall&&value<c.relativeTolerance*parameters->initial);
            recurrence[7]=recurrence[5]||((recurrence[6]>=c.maximumIterations||converged)&&recurrence[6]>=c.minimumIterations);
        }
        __syncthreads();
    }
    // Publish the actual FP32 b-A*x-affine residual, not its recurrence.
    product=0;
    if(i<n){
        product=values[offsets[i]]*x[i];
        for(int e=offsets[i]+1;e<offsets[i+1];++e)product+=values[e]*x[columns[e]];
        if(affine)product+=affine[i];
    }
    sum=localBlockSum(i<n?fabsf(b[i]-product):0.0f,scratch);
    if(!i)recurrence[4]=sum;
    __syncthreads();
    if(!i){
        const auto& c=parameters->controls;const GpuReal value=recurrence[4]/parameters->norm;
        report->iterations=int(recurrence[6]);report->initialResidual=parameters->initial;report->finalResidual=value;
        report->singular=recurrence[5]!=0;report->converged=value<c.tolerance||
            (c.relativeTolerance>c.normalizationSmall&&value<c.relativeTolerance*parameters->initial);
    }
}

}
struct SparseConjugateGradient::Implementation {
    cudaStream_t stream=nullptr;PinnedTransferQueue transfers;
    GpuReal* hostRecurrence=nullptr;
    DeviceArray<int> offsets,columns,entrySource,entryKind,boundaryOffsets,boundaryIndices,boundaryNeighbour;
    DeviceArray<GpuReal> diagonal,upper,lower,source,boundaryDiagonal,boundarySource,boundaryJump,affine;
    std::vector<int> cachedOwner,cachedNeighbour,cachedBoundaryOwner,cachedBoundaryNeighbour;
    int cachedRows=-1,cachedEntries=0;
    std::vector<double> cachedBoundaryJump;
    SparseDeviceLduValues residentValues{};
    int residentRows=0;bool residentConfigured=false,residentAffine=false;
    DeviceArray<GpuReal> values,rowSum,x,b,r,z,p,ap,partial,result,recurrence;
    std::vector<GpuReal> interfaceX,interfaceResult,interfacePacked;
    std::vector<int> cachedInterfaceCells;
    DeviceArray<int> interfaceCells;
    DeviceArray<GpuReal> interfaceDevice;
    void configureInterface(int n,const SparseCommunication& comm){
        if(!comm.addInterface||comm.interfaceCellCount<0)return;
        const int count=comm.interfaceCellCount;
        if(count&&!comm.interfaceCells)throw std::invalid_argument("missing compact interface cells");
        std::vector<int> cells;if(count)cells.assign(comm.interfaceCells,comm.interfaceCells+count);
        std::sort(cells.begin(),cells.end());
        if((count&&(cells.front()<0||cells.back()>=n))||std::adjacent_find(cells.begin(),cells.end())!=cells.end())
            throw std::invalid_argument("invalid compact interface cells");
        if(cells!=cachedInterfaceCells){interfaceCells.upload(cells.data(),count,transfers);cachedInterfaceCells=std::move(cells);}
        interfacePacked.resize(2*count);interfaceDevice.reserve(2*count);
        interfaceX.resize(n);interfaceResult.resize(n);
    }
    DeviceArray<LocalSolveParameters> localParameters;
    DeviceArray<ConjugateGradientReport> localReport;
    ConjugateGradientControls graphControls{};
    struct Graph {
        cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
        void clear(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);executable=nullptr;graph=nullptr;}
        ~Graph(){clear();}
    } localGraph;
    int graphRows=-1;std::vector<const void*> graphAddresses;
    cudaEvent_t externalReady=nullptr,externalDone=nullptr;
    void waitFor(void* producer){
        if(!externalReady)checked(cudaEventCreateWithFlags(&externalReady,cudaEventDisableTiming));
        checked(cudaEventRecord(externalReady,static_cast<cudaStream_t>(producer)));
        checked(cudaStreamWaitEvent(stream,externalReady,0));
    }
    void publishTo(void* consumer){
        if(!externalDone)checked(cudaEventCreateWithFlags(&externalDone,cudaEventDisableTiming));
        checked(cudaEventRecord(externalDone,stream));
        checked(cudaStreamWaitEvent(static_cast<cudaStream_t>(consumer),externalDone,0));
    }
    Implementation(){checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfers.bind(stream);const auto status=cudaMallocHost(reinterpret_cast<void**>(&hostRecurrence),8*sizeof(GpuReal));if(status!=cudaSuccess){cudaStreamDestroy(stream);stream=nullptr;checked(status);}}
    ~Implementation(){transfers.discard();if(stream){cudaStreamSynchronize(stream);localGraph.clear();cudaStreamDestroy(stream);}if(hostRecurrence)cudaFreeHost(hostRecurrence);if(externalReady)cudaEventDestroy(externalReady);if(externalDone)cudaEventDestroy(externalDone);}
    void reduce(int n,int mode,const GpuReal* a,const GpuReal* other,GpuReal* output,
        const SparseCommunication& comm,GpuReal mean=0,const GpuReal* stop=nullptr,const GpuReal* residentMean=nullptr){
        const int blocks=std::max(1,std::min(1024,(n+threads-1)/threads));partial.reserve(blocks);
        partialReduction<<<blocks,threads,0,stream>>>(n,mode,a,other,rowSum.ptr,mean,partial.ptr,stop,residentMean);
        finishReduction<<<1,threads,0,stream>>>(blocks,partial.ptr,output,stop);
        if(comm.globalSum){
            GpuReal host=0;transfers.download(&host,output,sizeof(GpuReal));
            transfers.finish();comm.globalSum(comm.context,&host,1);
            transfers.upload(output,&host,sizeof(GpuReal));
        }
    }
    GpuReal sum(int n,int mode,const GpuReal* a,const GpuReal* other,const SparseCommunication& comm,GpuReal mean=0){
        result.reserve(1);reduce(n,mode,a,other,result.ptr,comm,mean);
        GpuReal host=0;transfers.download(&host,result.ptr,sizeof(GpuReal));transfers.finish();return host;
    }
    void matvec(int n,const GpuReal* input,GpuReal* output,const SparseCommunication& comm,const GpuReal* stop=nullptr){
        if(n)multiply<<<(n+threads-1)/threads,threads,0,stream>>>(n,offsets.ptr,columns.ptr,values.ptr,input,output,stop);
        if(comm.addInterface){
            if(comm.interfaceCellCount>=0){
                const int count=int(cachedInterfaceCells.size());
                if(count){gatherInterface<<<(count+threads-1)/threads,threads,0,stream>>>(count,interfaceCells.ptr,input,output,interfaceDevice.ptr);
                    transfers.download(interfacePacked.data(),interfaceDevice.ptr,2*count*sizeof(GpuReal));}
                transfers.finish();
                for(int i=0;i<count;++i){const int c=cachedInterfaceCells[i];interfaceX[c]=interfacePacked[i];interfaceResult[c]=interfacePacked[count+i];}
                comm.addInterface(comm.context,interfaceX.data(),interfaceResult.data(),n);
                for(int i=0;i<count;++i)interfacePacked[i]=interfaceResult[cachedInterfaceCells[i]];
                if(count){transfers.upload(interfaceDevice.ptr,interfacePacked.data(),count*sizeof(GpuReal));
                    scatterInterface<<<(count+threads-1)/threads,threads,0,stream>>>(count,interfaceCells.ptr,interfaceDevice.ptr,output);}
            }else{
                interfaceX.resize(n);interfaceResult.resize(n);
                if(n){transfers.download(interfaceX.data(),input,n*sizeof(GpuReal));
                    transfers.download(interfaceResult.data(),output,n*sizeof(GpuReal));}
                transfers.finish();
                comm.addInterface(comm.context,interfaceX.data(),interfaceResult.data(),n);
                if(n)transfers.upload(output,interfaceResult.data(),n*sizeof(GpuReal));
            }
        }
    }
    void iteration(int n,GpuReal norm,GpuReal singularSmall,const SparseCommunication& comm,bool local=false){
        const GpuReal* stop=local?recurrence.ptr:nullptr;
        const int blocks=(n+threads-1)/threads;
        if(n)precondition<<<blocks,threads,0,stream>>>(n,offsets.ptr,values.ptr,r.ptr,z.ptr,stop);
        reduce(n,2,z.ptr,r.ptr,recurrence.ptr,comm,0,stop);
        if(n)direction<<<blocks,threads,0,stream>>>(n,z.ptr,recurrence.ptr,p.ptr);
        matvec(n,p.ptr,ap.ptr,comm,stop);
        reduce(n,2,ap.ptr,p.ptr,recurrence.ptr+1,comm,0,stop);
        prepareStep<<<1,1,0,stream>>>(recurrence.ptr,norm,singularSmall,local?localParameters.ptr:nullptr);
        if(n)update<<<blocks,threads,0,stream>>>(n,recurrence.ptr,p.ptr,ap.ptr,x.ptr,r.ptr);
        reduce(n,1,r.ptr,nullptr,recurrence.ptr+4,comm,0,stop);
    }


    cudaGraphNode_t appendLocal(cudaGraph_t graph,const std::vector<cudaGraphNode_t>& dependencies,
        int n,const ConjugateGradientControls& controls,const GpuReal* initialAffine,const ConjugateGradientControls* deviceControls=nullptr){
        using executionGraphDetail::appendCaptured;using executionGraphDetail::addWhile;
        if(n<=threads)return appendCaptured(graph,dependencies,stream,[&]{
            solveLocalBlock<<<1,threads,0,stream>>>(n,offsets.ptr,columns.ptr,values.ptr,rowSum.ptr,b.ptr,
                initialAffine,x.ptr,r.ptr,z.ptr,p.ptr,ap.ptr,controls,deviceControls,localParameters.ptr,recurrence.ptr,localReport.ptr);
        });
        cudaGraphConditionalHandle handle;
        checked(cudaGraphConditionalHandleCreate(&handle,graph,0,cudaGraphCondAssignDefault));
        const auto initialized=appendCaptured(graph,dependencies,stream,[&]{
            initializeLocalParameters<<<1,1,0,stream>>>(n,controls,localParameters.ptr,recurrence.ptr,deviceControls);
            matvec(n,x.ptr,ap.ptr,{});
            if(n&&initialAffine)addInitialAffine<<<(n+threads-1)/threads,threads,0,stream>>>(n,initialAffine,ap.ptr);
            if(n)initialResidual<<<(n+threads-1)/threads,threads,0,stream>>>(n,b.ptr,ap.ptr,r.ptr);
            reduce(n,0,x.ptr,nullptr,&localParameters.ptr->mean,{});
            normalizeLocalMean<<<1,1,0,stream>>>(n,&localParameters.ptr->mean);
            reduce(n,3,ap.ptr,b.ptr,&localParameters.ptr->norm,{},0,nullptr,&localParameters.ptr->mean);
            reduce(n,1,r.ptr,nullptr,recurrence.ptr+4,{});
            initializeLocalCondition<<<1,1,0,stream>>>(localParameters.ptr,recurrence.ptr,handle);
            const int blocks=std::max(1,std::min(1024,(n+threads-1)/threads));
            preconditionAndDot<<<blocks,threads,0,stream>>>(n,offsets.ptr,values.ptr,r.ptr,z.ptr,partial.ptr,recurrence.ptr);
            finishLocalReduction<0><<<1,threads,0,stream>>>(blocks,partial.ptr,recurrence.ptr,localParameters.ptr,handle);
        });
        cudaGraph_t body=nullptr;const auto loop=addWhile(graph,handle,{initialized},body);
        appendCaptured(body,{},stream,[&]{
            // No inter-rank callbacks are involved in this resident loop.
            // MPI/interface solves keep their existing exchange ordering below.
            const int blocks=std::max(1,std::min(1024,(n+threads-1)/threads));
            // Amortize conditional-graph loop dispatch without changing a
            // single physical iteration. Every stage observes the resident
            // stop flag; unused stages after convergence perform no arithmetic.
            for(int batch=0;batch<4;++batch){
                if(n)direction<<<(n+threads-1)/threads,threads,0,stream>>>(n,z.ptr,recurrence.ptr,p.ptr);
                multiplyAndDot<<<blocks,threads,0,stream>>>(n,offsets.ptr,columns.ptr,values.ptr,p.ptr,ap.ptr,partial.ptr,recurrence.ptr);
                finishLocalReduction<1><<<1,threads,0,stream>>>(blocks,partial.ptr,recurrence.ptr,localParameters.ptr,handle);
                updatePreconditionAndNorm<<<blocks,threads,0,stream>>>(n,offsets.ptr,values.ptr,recurrence.ptr,p.ptr,ap.ptr,x.ptr,r.ptr,z.ptr,partial.ptr);
                finishLocalReduction<2><<<1,threads,0,stream>>>(blocks,partial.ptr,recurrence.ptr,localParameters.ptr,handle);
            }
        });
        return appendCaptured(graph,{loop},stream,[&]{
            matvec(n,x.ptr,ap.ptr,{});
            if(n&&initialAffine)addInitialAffine<<<(n+threads-1)/threads,threads,0,stream>>>(n,initialAffine,ap.ptr);
            if(n)initialResidual<<<(n+threads-1)/threads,threads,0,stream>>>(n,b.ptr,ap.ptr,r.ptr);
            reduce(n,1,r.ptr,nullptr,recurrence.ptr+4,{});
            publishLocalReport<<<1,1,0,stream>>>(localParameters.ptr,recurrence.ptr,localReport.ptr);
        });
    }
    void prepareResidentRhs(GpuReal* solution){
        const int n=residentRows;const auto& values=residentValues;
        if(n){
            assembleLdu<<<(n+threads-1)/threads,threads,0,stream>>>(n,offsets.ptr,entrySource.ptr,entryKind.ptr,
                boundaryOffsets.ptr,boundaryIndices.ptr,boundaryNeighbour.ptr,values.diagonal,values.upper,values.lower,
                values.source,values.boundaryDiagonal,values.boundarySource,this->values.ptr,b.ptr,rowSum.ptr,
                residentAffine?(values.boundaryJump?values.boundaryJump:boundaryJump.ptr):nullptr,residentAffine?affine.ptr:nullptr);
            checked(cudaMemcpyAsync(x.ptr,solution,n*sizeof(GpuReal),cudaMemcpyDeviceToDevice,stream));
        }
    }
    void enqueueLocal(int n,const ConjugateGradientControls& controls,const GpuReal* initialAffine,bool launch=true){
        localParameters.reserve(1);localReport.reserve(1);recurrence.reserve(8);
        partial.reserve(2*std::max(1,std::min(1024,(n+threads-1)/threads)));
        const std::vector<const void*> addresses={offsets.ptr,columns.ptr,values.ptr,rowSum.ptr,
            r.ptr,z.ptr,p.ptr,ap.ptr,x.ptr,b.ptr,recurrence.ptr,partial.ptr,localParameters.ptr,localReport.ptr,initialAffine};
        const bool sameControls=graphControls.maximumIterations==controls.maximumIterations
            &&graphControls.minimumIterations==controls.minimumIterations&&graphControls.tolerance==controls.tolerance
            &&graphControls.relativeTolerance==controls.relativeTolerance&&graphControls.normalizationSmall==controls.normalizationSmall
            &&graphControls.singularSmall==controls.singularSmall;
        auto& graph=localGraph;
        if(graphRows!=n||graphAddresses!=addresses||!sameControls){
            graph.clear();graphRows=n;graphAddresses=addresses;graphControls=controls;
        }
        if(!graph.executable){
            checked(cudaGraphCreate(&graph.graph,0));
            appendLocal(graph.graph,{},n,controls,initialAffine);
            checked(cudaGraphInstantiateWithFlags(&graph.executable,graph.graph,0));
        }
        if(launch)checked(cudaGraphLaunch(graph.executable,stream));
    }

};
SparseConjugateGradient::SparseConjugateGradient():data_(new Implementation){}
SparseConjugateGradient::~SparseConjugateGradient()=default;
ExecutionGraphSignature SparseConjugateGradient::residentExecutionSignature()const{
    const auto& d=*data_;ExecutionGraphSignature key;key.add(this);
    for(const void* p:{
        static_cast<const void*>(d.offsets.ptr),static_cast<const void*>(d.columns.ptr),static_cast<const void*>(d.entrySource.ptr),
        static_cast<const void*>(d.entryKind.ptr),static_cast<const void*>(d.boundaryOffsets.ptr),static_cast<const void*>(d.boundaryIndices.ptr),
        static_cast<const void*>(d.boundaryNeighbour.ptr),static_cast<const void*>(d.boundaryJump.ptr),static_cast<const void*>(d.affine.ptr),
        static_cast<const void*>(d.values.ptr),static_cast<const void*>(d.rowSum.ptr),static_cast<const void*>(d.x.ptr),
        static_cast<const void*>(d.b.ptr),static_cast<const void*>(d.r.ptr),static_cast<const void*>(d.z.ptr),
        static_cast<const void*>(d.p.ptr),static_cast<const void*>(d.ap.ptr),static_cast<const void*>(d.partial.ptr),
        static_cast<const void*>(d.recurrence.ptr),static_cast<const void*>(d.localParameters.ptr),static_cast<const void*>(d.localReport.ptr)
    })key.add(p);
    key.add(d.residentRows);key.add(d.residentConfigured);key.add(d.residentAffine);
    const auto& v=d.residentValues;
    key.add(v.diagonal);key.add(v.upper);key.add(v.lower);key.add(v.source);
    key.add(v.boundaryDiagonal);key.add(v.boundarySource);key.add(v.boundaryJump);
    const auto& c=d.graphControls;
    key.add(c.maximumIterations);key.add(c.minimumIterations);key.add(c.tolerance);
    key.add(c.relativeTolerance);key.add(c.normalizationSmall);key.add(c.singularSmall);
    return key;
}
ConjugateGradientReport SparseConjugateGradient::solve(const SparseMatrixView& matrix,
    const double* source,double* solution,const ConjugateGradientControls& controls,const SparseCommunication& comm){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);const int n=matrix.rows;
    d.offsets.upload(matrix.offsets,n+1,d.transfers);d.columns.upload(matrix.columns,matrix.entries,d.transfers);
    d.values.upload(matrix.values,matrix.entries,d.transfers);d.rowSum.upload(matrix.rowSum,n,d.transfers);
    d.b.upload(source,n,d.transfers);
    // Raw CSR uploads replace the address storage, invalidating an LDU cache.
    d.cachedRows=-1;d.residentConfigured=false;
    return solvePrepared(n,solution,controls,comm);
}
ConjugateGradientReport SparseConjugateGradient::solvePrepared(int n,double* solution,
    const ConjugateGradientControls& controls,const SparseCommunication& comm,const GpuReal* initialAffine,GpuReal* deviceSolution){
    auto& d=*data_;d.configureInterface(n,comm);PinnedTransferQueue::Operation operation(d.transfers);const int blocks=(n+threads-1)/threads;
    if(deviceSolution){d.x.reserve(n);if(n)checked(cudaMemcpyAsync(d.x.ptr,deviceSolution,n*sizeof(GpuReal),cudaMemcpyDeviceToDevice,d.stream));}
    else d.x.upload(solution,n,d.transfers);
    d.r.reserve(n);d.z.reserve(n);d.p.reserve(n);d.ap.reserve(n);
    if(!comm.addInterface&&!comm.globalSum){
        d.enqueueLocal(n,controls,initialAffine);
        ConjugateGradientReport report;
        d.transfers.download(&report,d.localReport.ptr,sizeof(report));
        if(n){if(deviceSolution)checked(cudaMemcpyAsync(deviceSolution,d.x.ptr,n*sizeof(GpuReal),cudaMemcpyDeviceToDevice,d.stream));
            else d.transfers.downloadConverted(solution,d.x.ptr,n);}
        d.transfers.finish();return report;
    }
    d.matvec(n,d.x.ptr,d.ap.ptr,comm);
    if(n&&initialAffine)addInitialAffine<<<blocks,threads,0,d.stream>>>(n,initialAffine,d.ap.ptr);
    if(n)initialResidual<<<blocks,threads,0,d.stream>>>(n,d.b.ptr,d.ap.ptr,d.r.ptr);
    GpuReal count=n;if(comm.globalSum)comm.globalSum(comm.context,&count,1);
    const GpuReal mean=d.sum(n,0,d.x.ptr,nullptr,comm)/(count?count:1);
    const GpuReal norm=d.sum(n,3,d.ap.ptr,d.b.ptr,comm,mean)+controls.normalizationSmall;
    ConjugateGradientReport report;
    report.initialResidual=report.finalResidual=d.sum(n,1,d.r.ptr,nullptr,comm)/norm;
    auto converged=[&]{return report.finalResidual<controls.tolerance ||
        (controls.relativeTolerance>controls.normalizationSmall && report.finalResidual<controls.relativeTolerance*report.initialResidual);};
    report.converged=converged();
    if(controls.minimumIterations>0||!report.converged){
        d.recurrence.reserve(8);checked(cudaMemsetAsync(d.recurrence.ptr,0,8*sizeof(GpuReal),d.stream));
        do {
            d.iteration(n,norm,controls.singularSmall,comm);
            checked(cudaMemcpyAsync(d.hostRecurrence,d.recurrence.ptr,8*sizeof(GpuReal),cudaMemcpyDeviceToHost,d.stream));
            d.transfers.finish();report.iterations=int(d.hostRecurrence[6]);
            report.singular=d.hostRecurrence[5]!=0;
            report.finalResidual=d.hostRecurrence[4]/norm;report.converged=converged();
            if(report.singular)break;
        }while((report.iterations<controls.maximumIterations&&!report.converged)||report.iterations<controls.minimumIterations);
    }
    d.matvec(n,d.x.ptr,d.ap.ptr,comm);
    if(n&&initialAffine)addInitialAffine<<<blocks,threads,0,d.stream>>>(n,initialAffine,d.ap.ptr);
    if(n)initialResidual<<<blocks,threads,0,d.stream>>>(n,d.b.ptr,d.ap.ptr,d.r.ptr);
    report.finalResidual=d.sum(n,1,d.r.ptr,nullptr,comm)/norm;report.converged=converged();
    if(n){if(deviceSolution)checked(cudaMemcpyAsync(deviceSolution,d.x.ptr,n*sizeof(GpuReal),cudaMemcpyDeviceToDevice,d.stream));
        else d.transfers.downloadConverted(solution,d.x.ptr,n);}
    d.transfers.finish();return report;
}
void SparseConjugateGradient::prepareLdu(const SparseLduView& matrix,const SparseDeviceLduValues* resident,bool assemble){
    auto& d=*data_;const int n=matrix.rows,nf=matrix.faces,nb=matrix.boundaryFaces;
    if(resident)d.waitFor(resident->readyStream);
    const auto same=[](const std::vector<int>& old,const int* values,int count){
        return int(old.size())==count&&(count==0||std::equal(old.begin(),old.end(),values));
    };
    if(d.cachedRows!=n||!same(d.cachedOwner,matrix.owner,nf)||!same(d.cachedNeighbour,matrix.neighbour,nf)
        ||!same(d.cachedBoundaryOwner,matrix.boundaryOwner,nb)||!same(d.cachedBoundaryNeighbour,matrix.boundaryNeighbour,nb)){
        struct Entry {int column,source,kind;};std::vector<std::vector<Entry>> rows(n);
        std::vector<std::vector<int>> boundaries(n);
        for(int f=0;f<nf;++f){rows[matrix.neighbour[f]].push_back({matrix.owner[f],f,1});rows[matrix.owner[f]].push_back({matrix.neighbour[f],f,2});}
        for(int f=0;f<nb;++f){
            const int own=matrix.boundaryOwner[f],neighbour=matrix.boundaryNeighbour[f];boundaries[own].push_back(f);
            if(neighbour>=0)rows[own].push_back({neighbour,f,3});
        }
        std::vector<int> offsets(1,0),columns,indices,kinds,bOffsets(1,0),bIndices;
        for(int c=0;c<n;++c){
            columns.push_back(c);indices.push_back(c);kinds.push_back(0);
            for(const auto& entry:rows[c]){columns.push_back(entry.column);indices.push_back(entry.source);kinds.push_back(entry.kind);}
            offsets.push_back(columns.size());bIndices.insert(bIndices.end(),boundaries[c].begin(),boundaries[c].end());bOffsets.push_back(bIndices.size());
        }
        d.offsets.upload(offsets.data(),offsets.size(),d.transfers);d.columns.upload(columns.data(),columns.size(),d.transfers);
        d.entrySource.upload(indices.data(),indices.size(),d.transfers);d.entryKind.upload(kinds.data(),kinds.size(),d.transfers);
        d.boundaryOffsets.upload(bOffsets.data(),bOffsets.size(),d.transfers);d.boundaryIndices.upload(bIndices.data(),bIndices.size(),d.transfers);
        d.boundaryNeighbour.upload(matrix.boundaryNeighbour,nb,d.transfers);
        // Complete async use of temporary topology buffers before releasing them.
        d.transfers.finish();
        const auto copy=[](std::vector<int>& out,const int* input,int count){out.clear();if(count)out.assign(input,input+count);};
        copy(d.cachedOwner,matrix.owner,nf);copy(d.cachedNeighbour,matrix.neighbour,nf);
        copy(d.cachedBoundaryOwner,matrix.boundaryOwner,nb);copy(d.cachedBoundaryNeighbour,matrix.boundaryNeighbour,nb);
        d.cachedRows=n;d.cachedEntries=columns.size();
    }
    if(!resident){
        d.diagonal.upload(matrix.diagonal,n,d.transfers);d.upper.upload(matrix.upper,nf,d.transfers);d.lower.upload(matrix.lower,nf,d.transfers);
        d.source.upload(matrix.source,n,d.transfers);d.boundaryDiagonal.upload(matrix.boundaryDiagonal,nb,d.transfers);d.boundarySource.upload(matrix.boundarySource,nb,d.transfers);
    }
    if(matrix.boundaryJump){
        if(d.cachedBoundaryJump.size()!=std::size_t(nb)||
            (nb&&std::memcmp(d.cachedBoundaryJump.data(),matrix.boundaryJump,nb*sizeof(double)))){
            d.boundaryJump.upload(matrix.boundaryJump,nb,d.transfers);
            d.cachedBoundaryJump.assign(matrix.boundaryJump,matrix.boundaryJump+nb);
        }
        d.affine.reserve(n);
    }
    d.values.reserve(d.cachedEntries);d.b.reserve(n);d.rowSum.reserve(n);
    const SparseDeviceLduValues values=resident?*resident:SparseDeviceLduValues{
        d.diagonal.ptr,d.upper.ptr,d.lower.ptr,d.source.ptr,d.boundaryDiagonal.ptr,d.boundarySource.ptr,d.stream};
    if(n&&assemble)assembleLdu<<<(n+threads-1)/threads,threads,0,d.stream>>>(n,d.offsets.ptr,d.entrySource.ptr,d.entryKind.ptr,
        d.boundaryOffsets.ptr,d.boundaryIndices.ptr,d.boundaryNeighbour.ptr,values.diagonal,values.upper,values.lower,
        values.source,values.boundaryDiagonal,values.boundarySource,d.values.ptr,d.b.ptr,d.rowSum.ptr,
        matrix.boundaryJump?(values.boundaryJump?values.boundaryJump:d.boundaryJump.ptr):nullptr,matrix.boundaryJump?d.affine.ptr:nullptr);
}
ConjugateGradientReport SparseConjugateGradient::solveLdu(const SparseLduView& matrix,double* solution,
    const ConjugateGradientControls& controls,const SparseCommunication& comm){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);d.residentConfigured=false;prepareLdu(matrix,nullptr);
    return solvePrepared(matrix.rows,solution,controls,comm,matrix.boundaryJump?d.affine.ptr:nullptr);
}
ConjugateGradientReport SparseConjugateGradient::solveResidentLdu(const SparseLduView& matrix,
    const SparseDeviceLduValues& values,GpuReal* solution,const ConjugateGradientControls& controls,const SparseCommunication& comm){
    if(!comm.addInterface&&!comm.globalSum){
        configureResidentLocalLdu(matrix,values,controls);enqueueResidentLocalLdu(solution);
        return collectResidentLocalReport();
    }
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);d.residentConfigured=false;prepareLdu(matrix,&values);
    const auto report=solvePrepared(matrix.rows,nullptr,controls,comm,matrix.boundaryJump?d.affine.ptr:nullptr,solution);
    d.publishTo(values.readyStream);return report;
}
void SparseConjugateGradient::configureResidentLocalLdu(const SparseLduView& matrix,
    const SparseDeviceLduValues& values,const ConjugateGradientControls& controls){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);prepareLdu(matrix,&values,false);
    const int n=matrix.rows;d.x.reserve(n);d.r.reserve(n);d.z.reserve(n);d.p.reserve(n);d.ap.reserve(n);
    d.residentRows=n;d.residentValues=values;d.residentAffine=matrix.boundaryJump!=nullptr;
    d.enqueueLocal(n,controls,d.residentAffine?d.affine.ptr:nullptr,false);d.residentConfigured=true;
}
ConjugateGradientDeviceReport SparseConjugateGradient::enqueueResidentLocalLdu(GpuReal* solution){
    auto& d=*data_;if(!d.residentConfigured)throw std::logic_error("resident local sparse solve is not configured");
    const int n=d.residentRows;const auto& values=d.residentValues;d.waitFor(values.readyStream);
    d.prepareResidentRhs(solution);
    checked(cudaGraphLaunch(d.localGraph.executable,d.stream));
    if(n)checked(cudaMemcpyAsync(solution,d.x.ptr,n*sizeof(GpuReal),cudaMemcpyDeviceToDevice,d.stream));
    d.publishTo(values.readyStream);return {d.localReport.ptr,d.stream};
}
ExecutionGraphNode SparseConjugateGradient::appendResidentLocalLdu(ExecutionGraph graph,ExecutionGraphNode dependency,GpuReal* solution,const ConjugateGradientControls* deviceControls){
    auto& d=*data_;if(!d.residentConfigured)throw std::logic_error("resident local sparse solve is not configured");
    using executionGraphDetail::appendCaptured;
    std::vector<cudaGraphNode_t> dependencies;if(dependency)dependencies.push_back(dependency);
    const auto prepared=appendCaptured(graph,dependencies,d.stream,[&]{d.prepareResidentRhs(solution);});
    const auto solved=d.appendLocal(graph,{prepared},d.residentRows,d.graphControls,d.residentAffine?d.affine.ptr:nullptr,deviceControls);
    return appendCaptured(graph,{solved},d.stream,[&]{
        if(d.residentRows)checked(cudaMemcpyAsync(solution,d.x.ptr,d.residentRows*sizeof(GpuReal),cudaMemcpyDeviceToDevice,d.stream));
    });
}
ExecutionGraphNode SparseConjugateGradient::appendResidentLocalResidual(ExecutionGraph graph,ExecutionGraphNode dependency,GpuReal* solution,GpuReal* residual){
    auto& d=*data_;if(!d.residentConfigured)throw std::logic_error("resident local sparse residual is not configured");
    return executionGraphDetail::appendCaptured(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},d.stream,[&]{
        d.prepareResidentRhs(solution);const int n=d.residentRows;d.matvec(n,d.x.ptr,d.ap.ptr,{});
        if(n&&d.residentAffine)addInitialAffine<<<(n+threads-1)/threads,threads,0,d.stream>>>(n,d.affine.ptr,d.ap.ptr);
        if(n)initialResidual<<<(n+threads-1)/threads,threads,0,d.stream>>>(n,d.b.ptr,d.ap.ptr,residual);
    });
}
const ConjugateGradientReport* SparseConjugateGradient::deviceReport() const{return data_->localReport.ptr;}
ConjugateGradientReport SparseConjugateGradient::collectResidentLocalReport(){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);ConjugateGradientReport report;
    if(!d.residentConfigured)throw std::logic_error("resident local sparse report is not configured");
    d.transfers.download(&report,d.localReport.ptr,sizeof(report));d.transfers.finish();return report;
}
void SparseConjugateGradient::residualResidentLdu(const SparseLduView& matrix,
    const SparseDeviceLduValues& values,const GpuReal* solution,GpuReal* residual,const SparseCommunication& comm){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);prepareLdu(matrix,&values);
    const int n=matrix.rows;d.configureInterface(n,comm);d.ap.reserve(n);
    d.matvec(n,solution,d.ap.ptr,comm);
    if(n&&matrix.boundaryJump)addInitialAffine<<<(n+threads-1)/threads,threads,0,d.stream>>>(n,d.affine.ptr,d.ap.ptr);
    if(n)initialResidual<<<(n+threads-1)/threads,threads,0,d.stream>>>(n,d.b.ptr,d.ap.ptr,residual);
    d.publishTo(values.readyStream);
}

}}
