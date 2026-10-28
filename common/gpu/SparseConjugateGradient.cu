#include "SparseConjugateGradient.H"
#include <cuda_runtime.h>
#include "PinnedTransferQueue.cuh"
#include <algorithm>
#include <cmath>
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
__global__ void prepareStep(double* recurrence,double norm,double singularSmall){
    if(blockIdx.x||threadIdx.x||recurrence[7])return;
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
    const double* rowSum,double mean,double* partial,const double* stop){
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
__global__ void finishIteration(double* recurrence,double norm,double initial,
    ConjugateGradientControls controls,cudaGraphConditionalHandle handle){
    if(blockIdx.x||threadIdx.x||recurrence[7])return;
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
    DeviceArray<double> values,rowSum,x,b,r,z,p,ap,partial,result,recurrence;
    std::vector<double> interfaceX,interfaceResult;
    Implementation(){checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfers.bind(stream);const auto status=cudaMallocHost(reinterpret_cast<void**>(&hostRecurrence),8*sizeof(double));if(status!=cudaSuccess){cudaStreamDestroy(stream);stream=nullptr;checked(status);}}
    ~Implementation(){transfers.discard();if(stream){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}if(hostRecurrence)cudaFreeHost(hostRecurrence);}
    void reduce(int n,int mode,const double* a,const double* other,double* output,
        const SparseCommunication& comm,double mean=0,const double* stop=nullptr){
        const int blocks=std::max(1,std::min(1024,(n+threads-1)/threads));partial.reserve(blocks);
        partialReduction<<<blocks,threads,0,stream>>>(n,mode,a,other,rowSum.ptr,mean,partial.ptr,stop);
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
            interfaceX.resize(n);interfaceResult.resize(n);
            if(n){transfers.download(interfaceX.data(),input,n*sizeof(double));
                transfers.download(interfaceResult.data(),output,n*sizeof(double));}
            transfers.finish();
            comm.addInterface(comm.context,interfaceX.data(),interfaceResult.data(),n);
            if(n)transfers.upload(output,interfaceResult.data(),n*sizeof(double));
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
        prepareStep<<<1,1,0,stream>>>(recurrence.ptr,norm,singularSmall);
        if(n)update<<<blocks,threads,0,stream>>>(n,recurrence.ptr,p.ptr,ap.ptr,x.ptr,r.ptr);
        reduce(n,1,r.ptr,nullptr,recurrence.ptr+4,comm,0,stop);
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
    d.cachedRows=-1;
    return solvePrepared(n,solution,controls,comm);
}
ConjugateGradientReport SparseConjugateGradient::solvePrepared(int n,double* solution,
    const ConjugateGradientControls& controls,const SparseCommunication& comm,const double* initialAffine){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);const int blocks=(n+threads-1)/threads;
    d.x.upload(solution,n,d.transfers);
    d.r.reserve(n);d.z.reserve(n);d.p.reserve(n);d.ap.reserve(n);
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
        struct Graph {
            cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
            ~Graph(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);}
        } graph;
        // Local solves retain the unchanged recurrence and every stopping
        // decision inside a conditional CUDA graph. MPI/interface callbacks
        // use the same kernels around the native host communication contract.
        const bool local=!comm.addInterface&&!comm.globalSum;
        if(local){
            checked(cudaGraphCreate(&graph.graph,0));
            cudaGraphConditionalHandle handle;
            checked(cudaGraphConditionalHandleCreate(&handle,graph.graph,1,cudaGraphCondAssignDefault));
            cudaGraphNodeParams parameters{};parameters.type=cudaGraphNodeTypeConditional;
            parameters.conditional.handle=handle;parameters.conditional.type=cudaGraphCondTypeWhile;parameters.conditional.size=1;
            cudaGraphNode_t loop;
            checked(cudaGraphAddNode(&loop,graph.graph,nullptr,nullptr,0,&parameters));
            cudaGraph_t body=parameters.conditional.phGraph_out[0];
            checked(cudaStreamBeginCaptureToGraph(d.stream,body,nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal));
            d.iteration(n,norm,controls.singularSmall,comm,true);
            finishIteration<<<1,1,0,d.stream>>>(d.recurrence.ptr,norm,report.initialResidual,controls,handle);
            checked(cudaStreamEndCapture(d.stream,&body));
            cudaGraphNode_t notification;
            checked(cudaGraphAddMemcpyNode1D(&notification,graph.graph,&loop,1,d.hostRecurrence,d.recurrence.ptr,
                8*sizeof(double),cudaMemcpyDeviceToHost));
            checked(cudaGraphInstantiateWithFlags(&graph.executable,graph.graph,0));
        }
        do {
            if(local)checked(cudaGraphLaunch(graph.executable,d.stream));
            else {
                d.iteration(n,norm,controls.singularSmall,comm);
                checked(cudaMemcpyAsync(d.hostRecurrence,d.recurrence.ptr,8*sizeof(double),cudaMemcpyDeviceToHost,d.stream));
            }
            d.transfers.finish();report.iterations=int(d.hostRecurrence[6]);
            report.singular=d.hostRecurrence[5]!=0;
            report.finalResidual=d.hostRecurrence[4]/norm;report.converged=converged();
            if(report.singular)break;
        }while((report.iterations<controls.maximumIterations&&!report.converged)||report.iterations<controls.minimumIterations);
    }
    if(n)d.transfers.download(solution,d.x.ptr,n*sizeof(double));
    d.transfers.finish();return report;
}
ConjugateGradientReport SparseConjugateGradient::solveLdu(const SparseLduView& matrix,double* solution,
    const ConjugateGradientControls& controls,const SparseCommunication& comm){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);const int n=matrix.rows,nf=matrix.faces,nb=matrix.boundaryFaces;
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
    d.diagonal.upload(matrix.diagonal,n,d.transfers);d.upper.upload(matrix.upper,nf,d.transfers);d.lower.upload(matrix.lower,nf,d.transfers);
    d.source.upload(matrix.source,n,d.transfers);d.boundaryDiagonal.upload(matrix.boundaryDiagonal,nb,d.transfers);d.boundarySource.upload(matrix.boundarySource,nb,d.transfers);
    if(matrix.boundaryJump){d.boundaryJump.upload(matrix.boundaryJump,nb,d.transfers);d.affine.reserve(n);}
    d.values.reserve(d.cachedEntries);d.b.reserve(n);d.rowSum.reserve(n);
    if(n)assembleLdu<<<(n+threads-1)/threads,threads,0,d.stream>>>(n,d.offsets.ptr,d.entrySource.ptr,d.entryKind.ptr,
        d.boundaryOffsets.ptr,d.boundaryIndices.ptr,d.boundaryNeighbour.ptr,d.diagonal.ptr,d.upper.ptr,d.lower.ptr,
        d.source.ptr,d.boundaryDiagonal.ptr,d.boundarySource.ptr,d.values.ptr,d.b.ptr,d.rowSum.ptr,
        matrix.boundaryJump?d.boundaryJump.ptr:nullptr,matrix.boundaryJump?d.affine.ptr:nullptr);
    return solvePrepared(n,solution,controls,comm,matrix.boundaryJump?d.affine.ptr:nullptr);
}

}}
