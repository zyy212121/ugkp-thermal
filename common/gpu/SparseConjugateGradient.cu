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
    void upload(const T* host,std::size_t count,PinnedTransferQueue& queue){
        reserve(count);queue.upload(ptr,host,count*sizeof(T));
    }
};
constexpr int threads=256;
__global__ void multiply(int n,const int* offsets,const int* columns,const double* values,const double* x,double* y,const double* stop){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||(stop&&stop[7]))return;
    double value=values[offsets[i]]*x[i];
    for(int e=offsets[i]+1;e<offsets[i+1];++e)value+=values[e]*x[columns[e]];
    y[i]=value;
}
__global__ void assembleLdu(int n,const int* offsets,const int* entrySource,const int* entryKind,
    const int* boundaryOffsets,const int* boundaryIndices,const int* boundaryNeighbour,
    const double* diagonal,const double* upper,const double* lower,const double* source,
    const double* boundaryDiagonal,const double* boundarySource,double* values,double* rhs,double* rowSum,
    const double* boundaryJump=nullptr,double* affine=nullptr){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    double d=diagonal[i],b=source[i],jump=0;
    for(int j=boundaryOffsets[i];j<boundaryOffsets[i+1];++j){
        const int f=boundaryIndices[j];d+=boundaryDiagonal[f];
        if(boundaryNeighbour[f]==-1)b+=boundarySource[f];
        if(boundaryJump)jump+=boundarySource[f]*boundaryJump[f];
    }
    values[offsets[i]]=d;rhs[i]=b;double sum=d;
    for(int j=offsets[i]+1;j<offsets[i+1];++j){
        const int f=entrySource[j],kind=entryKind[j];
        const double value=kind==1?lower[f]:kind==2?upper[f]:-boundarySource[f];
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
__global__ void addInitialAffine(int n,const double* affine,double* product){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)product[i]+=affine[i];
}
__global__ void initialResidual(int n,const double* source,const double* ax,double* r){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)r[i]=source[i]-ax[i];
}
__global__ void precondition(int n,const int* offsets,const double* values,const double* r,double* z,const double* stop){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&!(stop&&stop[7]))z[i]=r[i]/values[offsets[i]];
}
// Recurrence scalars: rho, pAp, previous rho, alpha, |r|_1, singular, completed iterations.
__global__ void direction(int n,const double* z,const double* recurrence,double* p){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n&&!recurrence[7])p[i]=recurrence[6]==0?z[i]:z[i]+(recurrence[0]/recurrence[2])*p[i];
}
struct LocalSolveParameters {double norm,initial,mean;ConjugateGradientControls controls;};
__global__ void initializeLocalParameters(int count,ConjugateGradientControls controls,
    LocalSolveParameters* parameters,double* recurrence,const ConjugateGradientControls* deviceControls){
    if(blockIdx.x||threadIdx.x)return;
    *parameters={0,0,0,deviceControls?*deviceControls:controls};
    for(int i=0;i<8;++i)recurrence[i]=0;
}
__global__ void normalizeLocalMean(int count,double* mean){
    if(!blockIdx.x&&!threadIdx.x)*mean/=count?count:1;
}
__global__ void initializeLocalCondition(LocalSolveParameters* parameters,double* recurrence,
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
__global__ void publishLocalReport(const LocalSolveParameters* parameters,const double* recurrence,
    ConjugateGradientReport* report){
    if(blockIdx.x||threadIdx.x)return;
    const auto& c=parameters->controls;const double residual=recurrence[4]/parameters->norm;
    report->iterations=int(recurrence[6]);report->initialResidual=parameters->initial;report->finalResidual=residual;
    report->singular=recurrence[5]!=0;
    report->converged=residual<c.tolerance ||
        (c.relativeTolerance>c.normalizationSmall&&residual<c.relativeTolerance*parameters->initial);
}
__global__ void prepareStep(double* recurrence,double norm,double singularSmall,const LocalSolveParameters* parameters=nullptr){
    if(blockIdx.x||threadIdx.x||recurrence[7])return;
    if(parameters){norm=parameters->norm;singularSmall=parameters->controls.singularSmall;}
    recurrence[5]=fabs(recurrence[1])/norm<singularSmall;
    if(!recurrence[5]){recurrence[3]=recurrence[0]/recurrence[1];recurrence[6]+=1;}
    recurrence[2]=recurrence[0];
}
__global__ void update(int n,const double* recurrence,const double* p,const double* ap,double* x,double* r){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n&&!recurrence[5]&&!recurrence[7]){x[i]+=recurrence[3]*p[i];r[i]-=recurrence[3]*ap[i];}
}
// Fixed reduction tree; no floating-point atomics or scheduling-dependent sums.
// mode: sum, absolute sum, dot, finite-volume residual normalization.
__global__ void partialReduction(int n,int mode,const double* a,const double* b,
    const double* rowSum,double mean,double* partial,const double* stop,const double* residentMean=nullptr){
    if(residentMean)mean=*residentMean;
    if(stop&&stop[7])return;
    __shared__ double values[threads];double v=0;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
        if(mode==0)v+=a[i];
        else if(mode==1)v+=fabs(a[i]);
        else if(mode==2)v+=a[i]*b[i];
        else {const double reference=rowSum[i]*mean;v+=fabs(a[i]-reference)+fabs(b[i]-reference);}
    }
    values[threadIdx.x]=v;__syncthreads();
    for(int stride=threads/2;stride;stride/=2){if(threadIdx.x<stride)values[threadIdx.x]+=values[threadIdx.x+stride];__syncthreads();}
    if(threadIdx.x==0)partial[blockIdx.x]=values[0];
}
__global__ void finishReduction(int n,const double* partial,double* result,const double* stop){
    if(stop&&stop[7])return;
    __shared__ double values[threads];double v=0;
    for(int i=threadIdx.x;i<n;i+=threads)v+=partial[i];
    values[threadIdx.x]=v;__syncthreads();
    for(int stride=threads/2;stride;stride/=2){if(threadIdx.x<stride)values[threadIdx.x]+=values[threadIdx.x+stride];__syncthreads();}
    if(threadIdx.x==0)*result=values[0];
}
__global__ void gatherInterface(int count,const int* cells,const double* x,const double* y,double* packed){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){packed[i]=x[cells[i]];packed[count+i]=y[cells[i]];}
}
__global__ void scatterInterface(int count,const int* cells,const double* packed,double* y){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)y[cells[i]]=packed[i];
}
__global__ void finishIteration(double* recurrence,const LocalSolveParameters* parameters,cudaGraphConditionalHandle handle){
    if(blockIdx.x||threadIdx.x||recurrence[7])return;
    const auto controls=parameters->controls;const double norm=parameters->norm,initial=parameters->initial;
    const double residual=recurrence[4]/norm;
    const bool converged=residual<controls.tolerance ||
        (controls.relativeTolerance>controls.normalizationSmall&&residual<controls.relativeTolerance*initial);
    recurrence[7]=recurrence[5] ||
        ((recurrence[6]>=controls.maximumIterations||converged)&&recurrence[6]>=controls.minimumIterations);
    cudaGraphSetConditional(handle,recurrence[7]?0:1);
}
}
struct SparseConjugateGradient::Implementation {
    cudaStream_t stream=nullptr;PinnedTransferQueue transfers;
    double* hostRecurrence=nullptr;
    DeviceArray<int> offsets,columns,entrySource,entryKind,boundaryOffsets,boundaryIndices,boundaryNeighbour;
    DeviceArray<double> diagonal,upper,lower,source,boundaryDiagonal,boundarySource,boundaryJump,affine;
    std::vector<int> cachedOwner,cachedNeighbour,cachedBoundaryOwner,cachedBoundaryNeighbour;
    int cachedRows=-1,cachedEntries=0;
    std::vector<double> cachedBoundaryJump;
    SparseDeviceLduValues residentValues{};
    int residentRows=0;bool residentConfigured=false,residentAffine=false;
    DeviceArray<double> values,rowSum,x,b,r,z,p,ap,partial,result,recurrence;
    std::vector<double> interfaceX,interfaceResult,interfacePacked;
    std::vector<int> cachedInterfaceCells;
    DeviceArray<int> interfaceCells;
    DeviceArray<double> interfaceDevice;
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
    Implementation(){checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfers.bind(stream);const auto status=cudaMallocHost(reinterpret_cast<void**>(&hostRecurrence),8*sizeof(double));if(status!=cudaSuccess){cudaStreamDestroy(stream);stream=nullptr;checked(status);}}
    ~Implementation(){transfers.discard();if(stream){cudaStreamSynchronize(stream);localGraph.clear();cudaStreamDestroy(stream);}if(hostRecurrence)cudaFreeHost(hostRecurrence);if(externalReady)cudaEventDestroy(externalReady);if(externalDone)cudaEventDestroy(externalDone);}
    void reduce(int n,int mode,const double* a,const double* other,double* output,
        const SparseCommunication& comm,double mean=0,const double* stop=nullptr,const double* residentMean=nullptr){
        const int blocks=std::max(1,std::min(1024,(n+threads-1)/threads));partial.reserve(blocks);
        partialReduction<<<blocks,threads,0,stream>>>(n,mode,a,other,rowSum.ptr,mean,partial.ptr,stop,residentMean);
        finishReduction<<<1,threads,0,stream>>>(blocks,partial.ptr,output,stop);
        if(comm.globalSum){
            double host=0;transfers.download(&host,output,sizeof(double));
            transfers.finish();comm.globalSum(comm.context,&host,1);
            transfers.upload(output,&host,sizeof(double));
        }
    }
    double sum(int n,int mode,const double* a,const double* other,const SparseCommunication& comm,double mean=0){
        result.reserve(1);reduce(n,mode,a,other,result.ptr,comm,mean);
        double host=0;transfers.download(&host,result.ptr,sizeof(double));transfers.finish();return host;
    }
    void matvec(int n,const double* input,double* output,const SparseCommunication& comm,const double* stop=nullptr){
        if(n)multiply<<<(n+threads-1)/threads,threads,0,stream>>>(n,offsets.ptr,columns.ptr,values.ptr,input,output,stop);
        if(comm.addInterface){
            if(comm.interfaceCellCount>=0){
                const int count=int(cachedInterfaceCells.size());
                if(count){gatherInterface<<<(count+threads-1)/threads,threads,0,stream>>>(count,interfaceCells.ptr,input,output,interfaceDevice.ptr);
                    transfers.download(interfacePacked.data(),interfaceDevice.ptr,2*count*sizeof(double));}
                transfers.finish();
                for(int i=0;i<count;++i){const int c=cachedInterfaceCells[i];interfaceX[c]=interfacePacked[i];interfaceResult[c]=interfacePacked[count+i];}
                comm.addInterface(comm.context,interfaceX.data(),interfaceResult.data(),n);
                for(int i=0;i<count;++i)interfacePacked[i]=interfaceResult[cachedInterfaceCells[i]];
                if(count){transfers.upload(interfaceDevice.ptr,interfacePacked.data(),count*sizeof(double));
                    scatterInterface<<<(count+threads-1)/threads,threads,0,stream>>>(count,interfaceCells.ptr,interfaceDevice.ptr,output);}
            }else{
                interfaceX.resize(n);interfaceResult.resize(n);
                if(n){transfers.download(interfaceX.data(),input,n*sizeof(double));
                    transfers.download(interfaceResult.data(),output,n*sizeof(double));}
                transfers.finish();
                comm.addInterface(comm.context,interfaceX.data(),interfaceResult.data(),n);
                if(n)transfers.upload(output,interfaceResult.data(),n*sizeof(double));
            }
        }
    }
    void iteration(int n,double norm,double singularSmall,const SparseCommunication& comm,bool local=false){
        const double* stop=local?recurrence.ptr:nullptr;
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
        int n,const ConjugateGradientControls& controls,const double* initialAffine,const ConjugateGradientControls* deviceControls=nullptr){
        using executionGraphDetail::appendCaptured;using executionGraphDetail::addWhile;
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
        });
        cudaGraph_t body=nullptr;const auto loop=addWhile(graph,handle,{initialized},body);
        appendCaptured(body,{},stream,[&]{
            iteration(n,0,0,{},true);
            finishIteration<<<1,1,0,stream>>>(recurrence.ptr,localParameters.ptr,handle);
        });
        return appendCaptured(graph,{loop},stream,[&]{
            publishLocalReport<<<1,1,0,stream>>>(localParameters.ptr,recurrence.ptr,localReport.ptr);
        });
    }
    void prepareResidentRhs(double* solution){
        const int n=residentRows;const auto& values=residentValues;
        if(n){
            assembleLdu<<<(n+threads-1)/threads,threads,0,stream>>>(n,offsets.ptr,entrySource.ptr,entryKind.ptr,
                boundaryOffsets.ptr,boundaryIndices.ptr,boundaryNeighbour.ptr,values.diagonal,values.upper,values.lower,
                values.source,values.boundaryDiagonal,values.boundarySource,this->values.ptr,b.ptr,rowSum.ptr,
                residentAffine?(values.boundaryJump?values.boundaryJump:boundaryJump.ptr):nullptr,residentAffine?affine.ptr:nullptr);
            checked(cudaMemcpyAsync(x.ptr,solution,n*sizeof(double),cudaMemcpyDeviceToDevice,stream));
        }
    }
    void enqueueLocal(int n,const ConjugateGradientControls& controls,const double* initialAffine,bool launch=true){
        localParameters.reserve(1);localReport.reserve(1);recurrence.reserve(8);
        partial.reserve(std::max(1,std::min(1024,(n+threads-1)/threads)));
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
    const ConjugateGradientControls& controls,const SparseCommunication& comm,const double* initialAffine,double* deviceSolution){
    auto& d=*data_;d.configureInterface(n,comm);PinnedTransferQueue::Operation operation(d.transfers);const int blocks=(n+threads-1)/threads;
    if(deviceSolution){d.x.reserve(n);if(n)checked(cudaMemcpyAsync(d.x.ptr,deviceSolution,n*sizeof(double),cudaMemcpyDeviceToDevice,d.stream));}
    else d.x.upload(solution,n,d.transfers);
    d.r.reserve(n);d.z.reserve(n);d.p.reserve(n);d.ap.reserve(n);
    if(!comm.addInterface&&!comm.globalSum){
        d.enqueueLocal(n,controls,initialAffine);
        ConjugateGradientReport report;
        d.transfers.download(&report,d.localReport.ptr,sizeof(report));
        if(n){if(deviceSolution)checked(cudaMemcpyAsync(deviceSolution,d.x.ptr,n*sizeof(double),cudaMemcpyDeviceToDevice,d.stream));
            else d.transfers.download(solution,d.x.ptr,n*sizeof(double));}
        d.transfers.finish();return report;
    }
    d.matvec(n,d.x.ptr,d.ap.ptr,comm);
    if(n&&initialAffine)addInitialAffine<<<blocks,threads,0,d.stream>>>(n,initialAffine,d.ap.ptr);
    if(n)initialResidual<<<blocks,threads,0,d.stream>>>(n,d.b.ptr,d.ap.ptr,d.r.ptr);
    double count=n;if(comm.globalSum)comm.globalSum(comm.context,&count,1);
    const double mean=d.sum(n,0,d.x.ptr,nullptr,comm)/(count?count:1);
    const double norm=d.sum(n,3,d.ap.ptr,d.b.ptr,comm,mean)+controls.normalizationSmall;
    ConjugateGradientReport report;
    report.initialResidual=report.finalResidual=d.sum(n,1,d.r.ptr,nullptr,comm)/norm;
    auto converged=[&]{return report.finalResidual<controls.tolerance ||
        (controls.relativeTolerance>controls.normalizationSmall && report.finalResidual<controls.relativeTolerance*report.initialResidual);};
    report.converged=converged();
    if(controls.minimumIterations>0||!report.converged){
        d.recurrence.reserve(8);checked(cudaMemsetAsync(d.recurrence.ptr,0,8*sizeof(double),d.stream));
        do {
            d.iteration(n,norm,controls.singularSmall,comm);
            checked(cudaMemcpyAsync(d.hostRecurrence,d.recurrence.ptr,8*sizeof(double),cudaMemcpyDeviceToHost,d.stream));
            d.transfers.finish();report.iterations=int(d.hostRecurrence[6]);
            report.singular=d.hostRecurrence[5]!=0;
            report.finalResidual=d.hostRecurrence[4]/norm;report.converged=converged();
            if(report.singular)break;
        }while((report.iterations<controls.maximumIterations&&!report.converged)||report.iterations<controls.minimumIterations);
    }
    if(n){if(deviceSolution)checked(cudaMemcpyAsync(deviceSolution,d.x.ptr,n*sizeof(double),cudaMemcpyDeviceToDevice,d.stream));
        else d.transfers.download(solution,d.x.ptr,n*sizeof(double));}
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
    const SparseDeviceLduValues& values,double* solution,const ConjugateGradientControls& controls,const SparseCommunication& comm){
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
ConjugateGradientDeviceReport SparseConjugateGradient::enqueueResidentLocalLdu(double* solution){
    auto& d=*data_;if(!d.residentConfigured)throw std::logic_error("resident local sparse solve is not configured");
    const int n=d.residentRows;const auto& values=d.residentValues;d.waitFor(values.readyStream);
    d.prepareResidentRhs(solution);
    checked(cudaGraphLaunch(d.localGraph.executable,d.stream));
    if(n)checked(cudaMemcpyAsync(solution,d.x.ptr,n*sizeof(double),cudaMemcpyDeviceToDevice,d.stream));
    d.publishTo(values.readyStream);return {d.localReport.ptr,d.stream};
}
ExecutionGraphNode SparseConjugateGradient::appendResidentLocalLdu(ExecutionGraph graph,ExecutionGraphNode dependency,double* solution,const ConjugateGradientControls* deviceControls){
    auto& d=*data_;if(!d.residentConfigured)throw std::logic_error("resident local sparse solve is not configured");
    using executionGraphDetail::appendCaptured;
    std::vector<cudaGraphNode_t> dependencies;if(dependency)dependencies.push_back(dependency);
    const auto prepared=appendCaptured(graph,dependencies,d.stream,[&]{d.prepareResidentRhs(solution);});
    const auto solved=d.appendLocal(graph,{prepared},d.residentRows,d.graphControls,d.residentAffine?d.affine.ptr:nullptr,deviceControls);
    return appendCaptured(graph,{solved},d.stream,[&]{
        if(d.residentRows)checked(cudaMemcpyAsync(solution,d.x.ptr,d.residentRows*sizeof(double),cudaMemcpyDeviceToDevice,d.stream));
    });
}
ExecutionGraphNode SparseConjugateGradient::appendResidentLocalResidual(ExecutionGraph graph,ExecutionGraphNode dependency,double* solution,double* residual){
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
    const SparseDeviceLduValues& values,const double* solution,double* residual,const SparseCommunication& comm){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);prepareLdu(matrix,&values);
    const int n=matrix.rows;d.configureInterface(n,comm);d.ap.reserve(n);
    d.matvec(n,solution,d.ap.ptr,comm);
    if(n&&matrix.boundaryJump)addInitialAffine<<<(n+threads-1)/threads,threads,0,d.stream>>>(n,d.affine.ptr,d.ap.ptr);
    if(n)initialResidual<<<(n+threads-1)/threads,threads,0,d.stream>>>(n,d.b.ptr,d.ap.ptr,residual);
    d.publishTo(values.readyStream);
}

}}
