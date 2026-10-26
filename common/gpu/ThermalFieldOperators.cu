#include "ThermalFieldOperators.H"
#include <cuda_runtime.h>
#include "PinnedTransferQueue.cuh"
#include <stdexcept>
#include <string>
namespace ugkp { namespace gpu { namespace {
void checked(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(std::string("CUDA thermal fields: ")+cudaGetErrorString(code));}
template<class T> struct Array {
    T* ptr=nullptr;std::size_t capacity=0;
    ~Array(){if(ptr)cudaFree(ptr);}
    void reserve(std::size_t n){if(n<=capacity)return;T* next=nullptr;checked(cudaMalloc(reinterpret_cast<void**>(&next),n*sizeof(T)));if(ptr)cudaFree(ptr);ptr=next;capacity=n;}
    void upload(const std::vector<T>& input,PinnedTransferQueue& queue){reserve(input.size());queue.upload(ptr,input.data(),input.size()*sizeof(T));}
    void download(std::vector<T>& output,int n,PinnedTransferQueue& queue){output.resize(n);queue.download(output.data(),ptr,n*sizeof(T));}
};
__device__ double linear(double a,double b,double w,int kind){return kind==0?w*(a-b)+b:kind==1?w*a+(1-w)*b:a;}
__global__ void interpolateScalar(int n,const ScalarFaceInterpolation* input,bool harmonic,double* output){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const auto& f=input[i];
    output[i]=harmonic&&f.kind!=2?(f.owner>0&&f.neighbour>0?1/((1-f.weight)/f.owner+f.weight/f.neighbour):0):linear(f.owner,f.neighbour,f.weight,f.kind);
}
__global__ void interpolateDot(int n,const VectorFaceInterpolation* input,double* output){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const auto& f=input[i];
    const double x=linear(f.owner.x,f.neighbour.x,f.weight,f.kind),y=linear(f.owner.y,f.neighbour.y,f.weight,f.kind),z=linear(f.owner.z,f.neighbour.z,f.weight,f.kind);
    output[i]=f.direction.x*x+f.direction.y*y+f.direction.z*z;
}
__global__ void cellGradient(int n,const int* offsets,const int* faces,const int* signs,
    const double* value,const ThermalVector* area,const double* volume,ThermalVector* output){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;ThermalVector g;
    for(int j=offsets[c];j<offsets[c+1];++j){const int f=faces[j];const ThermalVector v{area[f].x*value[f],area[f].y*value[f],area[f].z*value[f]};
        if(signs[j]>0){g.x+=v.x;g.y+=v.y;g.z+=v.z;}else{g.x-=v.x;g.y-=v.y;g.z-=v.z;}}
    output[c]={g.x/volume[c],g.y/volume[c],g.z/volume[c]};
}
__global__ void boundaryGradient(int n,const GradientBoundaryCorrection* input,ThermalVector* output){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const auto& f=input[i];
    const ThermalVector normal{f.area.x/f.magnitude,f.area.y/f.magnitude,f.area.z/f.magnitude};
    const double delta=f.normalGradient-(normal.x*f.gradient.x+normal.y*f.gradient.y+normal.z*f.gradient.z);
    output[i]={f.gradient.x+normal.x*delta,f.gradient.y+normal.y*delta,f.gradient.z+normal.z*delta};
}
}
struct ThermalFieldOperators::Implementation {
    cudaStream_t stream=nullptr;PinnedTransferQueue transfers;int cells=-1,internal=-1;
    std::vector<int> owner,neighbour;Array<int> offsets,faces,signs;
    Array<ScalarFaceInterpolation> scalar;Array<VectorFaceInterpolation> vector;Array<GradientBoundaryCorrection> boundary;
    Array<double> faceValue,volume,scalarOutput;Array<ThermalVector> area,vectorOutput;
    Implementation(){checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfers.bind(stream);}
    ~Implementation(){transfers.discard();if(stream){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}}
};
ThermalFieldOperators::ThermalFieldOperators():data_(new Implementation){}
ThermalFieldOperators::~ThermalFieldOperators()=default;
void ThermalFieldOperators::interpolate(const std::vector<ScalarFaceInterpolation>& input,bool harmonic,std::vector<double>& output){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);int n=input.size();d.scalar.upload(input,d.transfers);d.scalarOutput.reserve(n);
    if(n)interpolateScalar<<<(n+127)/128,128,0,d.stream>>>(n,d.scalar.ptr,harmonic,d.scalarOutput.ptr);
    d.scalarOutput.download(output,n,d.transfers);d.transfers.finish();
}
void ThermalFieldOperators::dotInterpolate(const std::vector<VectorFaceInterpolation>& input,std::vector<double>& output){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);int n=input.size();d.vector.upload(input,d.transfers);d.scalarOutput.reserve(n);
    if(n)interpolateDot<<<(n+127)/128,128,0,d.stream>>>(n,d.vector.ptr,d.scalarOutput.ptr);
    d.scalarOutput.download(output,n,d.transfers);d.transfers.finish();
}
void ThermalFieldOperators::correctGradient(const std::vector<GradientBoundaryCorrection>& input,std::vector<ThermalVector>& output){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);int n=input.size();d.boundary.upload(input,d.transfers);d.vectorOutput.reserve(n);
    if(n)boundaryGradient<<<(n+127)/128,128,0,d.stream>>>(n,d.boundary.ptr,d.vectorOutput.ptr);
    d.vectorOutput.download(output,n,d.transfers);d.transfers.finish();
}
void ThermalFieldOperators::gradient(const GaussGradientInput& input,std::vector<ThermalVector>& output){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);const int n=input.cells,nf=input.owner.size(),ni=input.internalFaces;
    if(d.cells!=n||d.internal!=ni||d.owner!=input.owner||d.neighbour!=input.neighbour){
        std::vector<std::vector<std::pair<int,int>>> rows(n);
        for(int f=0;f<nf;++f){rows[input.owner[f]].push_back({f,1});if(f<ni)rows[input.neighbour[f]].push_back({f,-1});}
        std::vector<int> offsets(1,0),faces,signs;for(const auto& row:rows){for(auto e:row){faces.push_back(e.first);signs.push_back(e.second);}offsets.push_back(faces.size());}
        d.offsets.upload(offsets,d.transfers);d.faces.upload(faces,d.transfers);d.signs.upload(signs,d.transfers);d.transfers.finish();
        d.cells=n;d.internal=ni;d.owner=input.owner;d.neighbour=input.neighbour;
    }
    d.faceValue.upload(input.faceValue,d.transfers);d.volume.upload(input.volume,d.transfers);d.area.upload(input.area,d.transfers);d.vectorOutput.reserve(n);
    if(n)cellGradient<<<(n+127)/128,128,0,d.stream>>>(n,d.offsets.ptr,d.faces.ptr,d.signs.ptr,d.faceValue.ptr,d.area.ptr,d.volume.ptr,d.vectorOutput.ptr);
    d.vectorOutput.download(output,n,d.transfers);d.transfers.finish();
}
}}
