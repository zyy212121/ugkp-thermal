// Real CUDA execution prerequisite. This is not a CHMT physics validation.
#include <cuda_runtime.h>
#include <iostream>
__global__ void identifyNativeDevice(int* value){*value=42;}
int main(){
    int count=0;
    if(cudaGetDeviceCount(&count)!=cudaSuccess||count<1){std::cerr<<"BLOCKED: no usable native CUDA device\n";return 2;}
    cudaDeviceProp properties{};
    if(cudaGetDeviceProperties(&properties,0)!=cudaSuccess){std::cerr<<"BLOCKED: cannot query native CUDA device\n";return 2;}
    int* device=nullptr;int host=0;
    if(cudaMalloc(reinterpret_cast<void**>(&device),sizeof(int))!=cudaSuccess){std::cerr<<"BLOCKED: native CUDA allocation failed\n";return 2;}
    identifyNativeDevice<<<1,1>>>(device);
    const bool good=cudaGetLastError()==cudaSuccess&&cudaDeviceSynchronize()==cudaSuccess
        &&cudaMemcpy(&host,device,sizeof(int),cudaMemcpyDeviceToHost)==cudaSuccess&&host==42;
    cudaFree(device);
    if(!good){std::cerr<<"BLOCKED: native CUDA kernel/device copy failed\n";return 2;}
    std::cout<<"native_cuda_kernel=passed\ndevice="<<properties.name<<"\ncompute_capability="<<properties.major<<'.'<<properties.minor<<'\n';
}
