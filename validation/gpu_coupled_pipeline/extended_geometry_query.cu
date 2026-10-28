#include "gpu/ExtendedGeometryMath.H"
#include <cuda_runtime.h>
#include <stdexcept>
using ugkp::gpu::geometry::Point;
namespace {
void check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
__global__ void query(int count,const Point* input,double* output){const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)output[i]=ugkp::gpu::geometry::extended::triangleDistance(input[4*i],input[4*i+1],input[4*i+2],input[4*i+3]);}
}
void extendedGeometryQuery(int count,const Point* input,double* output){
    if(!count)return;Point* dInput=nullptr;double* dOutput=nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&dInput),count*4*sizeof(Point)));
    try {check(cudaMalloc(reinterpret_cast<void**>(&dOutput),count*sizeof(double)));
        check(cudaMemcpy(dInput,input,count*4*sizeof(Point),cudaMemcpyHostToDevice));
        query<<<(count+127)/128,128>>>(count,dInput,dOutput);check(cudaMemcpy(output,dOutput,count*sizeof(double),cudaMemcpyDeviceToHost));
    }catch(...){cudaFree(dInput);if(dOutput)cudaFree(dOutput);throw;}
    check(cudaFree(dInput));check(cudaFree(dOutput));
}
