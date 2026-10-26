#include "ImplicitDiffusion.H"
#include <cuda_runtime.h>
#include "PinnedTransferQueue.cuh"
#include <algorithm>
#include <stdexcept>
#include <string>
namespace ugkp { namespace gpu { namespace {
void checked(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(std::string("CUDA implicit diffusion: ")+cudaGetErrorString(code));}
template<class T> struct Array {
    T* ptr=nullptr;std::size_t capacity=0;
    ~Array(){if(ptr)cudaFree(ptr);}
    void reserve(std::size_t n){if(n<=capacity)return;T* next=nullptr;checked(cudaMalloc(reinterpret_cast<void**>(&next),n*sizeof(T)));if(ptr)cudaFree(ptr);ptr=next;capacity=n;}
    void upload(const std::vector<T>& input,PinnedTransferQueue& queue){reserve(input.size());queue.upload(ptr,input.data(),input.size()*sizeof(T));}
    void download(std::vector<T>& output,int n,PinnedTransferQueue& queue){output.resize(n);queue.download(output.data(),ptr,n*sizeof(T));}
};
__global__ void faceFactors(int nf,int ni,const double* gamma,const double* area,const double* delta,
    const double* gradientInternal,const double* gradientBoundary,const double* correction,bool corrected,
    double* upper,double* internal,double* boundary,double* flux){
    const int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=nf)return;
    const double g=gamma[f]*area[f];
    if(f<ni)upper[f]=delta[f]*g;
    else {const int b=f-ni;internal[b]=g*gradientInternal[b];boundary[b]=(-g)*gradientBoundary[b];}
    flux[f]=corrected?g*correction[f]:0;
}
__global__ void transientCells(int n,const double* volume,const double* capacity,const double* rhs,double dt,
    bool sourceFromCapacity,double* diagonal,double* source){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    const double weight=capacity[c]/dt;
    const double forcing=sourceFromCapacity?weight*rhs[c]:rhs[c];
    diagonal[c]=volume[c]*weight-diagonal[c];
    source[c]=(0-source[c])+volume[c]*forcing;
}
__global__ void negate(int n,double* field,bool subtractFromZero){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)field[i]=subtractFromZero?0-field[i]:-field[i];
}
__global__ void cellAssembly(int n,int ni,const int* offsets,const int* faces,const int* signs,
    const double* upper,const double* flux,const double* volume,bool corrected,double* diagonal,double* source){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    double d=0,sum=0;
    for(int i=offsets[c];i<offsets[c+1];++i){const int f=faces[i];
        if(f<ni)d-=upper[f];
        if(signs[i]>0)sum+=flux[f];else sum-=flux[f];
    }
    diagonal[c]=d;
    // Preserve the native surfaceIntegrate -> divide by V -> matrix source
    // integration, including its floating-point rounding; do not cancel V.
    source[c]=corrected?0-volume[c]*(sum/volume[c]):0;
}
}
struct ImplicitDiffusion::Implementation {
    cudaStream_t stream=nullptr;PinnedTransferQueue transfers;int cells=-1,internal=-1;
    std::vector<int> owner,neighbour;
    Array<int> offsets,faces,signs;
    Array<double> volume,gamma,area,delta,gradientInternal,gradientBoundary,correction;
    Array<double> capacity,rhs,diagonal,source,upper,lower,boundaryDiagonal,boundarySource,flux;
    Implementation(){checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfers.bind(stream);}
    ~Implementation(){transfers.discard();if(stream){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}}
};
ImplicitDiffusion::ImplicitDiffusion():data_(new Implementation){}
ImplicitDiffusion::~ImplicitDiffusion()=default;
void ImplicitDiffusion::assemble(const DiffusionInput& input,DiffusionMatrix& output){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);const int n=input.cells,ni=input.internalFaces,nf=input.owner.size(),nb=nf-ni;
    if(d.cells!=n||d.internal!=ni||d.owner!=input.owner||d.neighbour!=input.neighbour){
        std::vector<std::vector<std::pair<int,int>>> rows(n);
        for(int f=0;f<nf;++f){rows[input.owner[f]].push_back({f,1});if(f<ni)rows[input.neighbour[f]].push_back({f,-1});}
        std::vector<int> offsets(1,0),faces,signs;
        for(const auto& row:rows){for(auto entry:row){faces.push_back(entry.first);signs.push_back(entry.second);}offsets.push_back(faces.size());}
        d.offsets.upload(offsets,d.transfers);d.faces.upload(faces,d.transfers);d.signs.upload(signs,d.transfers);
        d.transfers.finish();d.cells=n;d.internal=ni;d.owner=input.owner;d.neighbour=input.neighbour;
    }
    d.volume.upload(input.volume,d.transfers);d.gamma.upload(input.gamma,d.transfers);d.area.upload(input.area,d.transfers);d.delta.upload(input.delta,d.transfers);
    d.gradientInternal.upload(input.gradientInternal,d.transfers);d.gradientBoundary.upload(input.gradientBoundary,d.transfers);d.correction.upload(input.correction,d.transfers);
    d.diagonal.reserve(n);d.source.reserve(n);d.upper.reserve(ni);d.boundaryDiagonal.reserve(nb);d.boundarySource.reserve(nb);d.flux.reserve(nf);
    if(nf)faceFactors<<<(nf+127)/128,128,0,d.stream>>>(nf,ni,d.gamma.ptr,d.area.ptr,d.delta.ptr,d.gradientInternal.ptr,d.gradientBoundary.ptr,d.correction.ptr,input.corrected,d.upper.ptr,d.boundaryDiagonal.ptr,d.boundarySource.ptr,d.flux.ptr);
    if(n)cellAssembly<<<(n+127)/128,128,0,d.stream>>>(n,ni,d.offsets.ptr,d.faces.ptr,d.signs.ptr,d.upper.ptr,d.flux.ptr,d.volume.ptr,input.corrected,d.diagonal.ptr,d.source.ptr);
    d.diagonal.download(output.diagonal,n,d.transfers);d.source.download(output.source,n,d.transfers);d.upper.download(output.upper,ni,d.transfers);
    d.boundaryDiagonal.download(output.boundaryDiagonal,nb,d.transfers);d.boundarySource.download(output.boundarySource,nb,d.transfers);d.flux.download(output.fluxCorrection,nf,d.transfers);
    d.transfers.finish();
}
void ImplicitDiffusion::transient(const std::vector<double>& volume,const std::vector<double>& capacity,
    const std::vector<double>& rhs,double dt,bool sourceFromCapacity,DiffusionMatrix& matrix){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);const int n=volume.size(),ni=matrix.upper.size(),nl=matrix.lower.size(),nb=matrix.boundaryDiagonal.size(),nf=matrix.fluxCorrection.size();
    d.volume.upload(volume,d.transfers);d.capacity.upload(capacity,d.transfers);d.rhs.upload(rhs,d.transfers);
    d.diagonal.upload(matrix.diagonal,d.transfers);d.source.upload(matrix.source,d.transfers);d.upper.upload(matrix.upper,d.transfers);d.lower.upload(matrix.lower,d.transfers);
    d.boundaryDiagonal.upload(matrix.boundaryDiagonal,d.transfers);d.boundarySource.upload(matrix.boundarySource,d.transfers);d.flux.upload(matrix.fluxCorrection,d.transfers);
    if(n)transientCells<<<(n+127)/128,128,0,d.stream>>>(n,d.volume.ptr,d.capacity.ptr,d.rhs.ptr,dt,sourceFromCapacity,d.diagonal.ptr,d.source.ptr);
    if(ni)negate<<<(ni+127)/128,128,0,d.stream>>>(ni,d.upper.ptr,false);
    if(nl)negate<<<(nl+127)/128,128,0,d.stream>>>(nl,d.lower.ptr,false);
    if(nb){negate<<<(nb+127)/128,128,0,d.stream>>>(nb,d.boundaryDiagonal.ptr,true);negate<<<(nb+127)/128,128,0,d.stream>>>(nb,d.boundarySource.ptr,true);}
    if(nf)negate<<<(nf+127)/128,128,0,d.stream>>>(nf,d.flux.ptr,false);
    d.diagonal.download(matrix.diagonal,n,d.transfers);d.source.download(matrix.source,n,d.transfers);d.upper.download(matrix.upper,ni,d.transfers);d.lower.download(matrix.lower,nl,d.transfers);
    d.boundaryDiagonal.download(matrix.boundaryDiagonal,nb,d.transfers);d.boundarySource.download(matrix.boundarySource,nb,d.transfers);d.flux.download(matrix.fluxCorrection,nf,d.transfers);
    d.transfers.finish();
}

}}
