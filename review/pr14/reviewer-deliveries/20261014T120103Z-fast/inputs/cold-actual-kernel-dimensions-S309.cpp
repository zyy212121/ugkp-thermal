#include <cuda_runtime_api.h>
#include <cuda.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t kernel,dim3 grid,dim3 block,void**args,std::size_t shared,cudaStream_t stream){
 using F=cudaError_t(*)(cudaKernel_t,dim3,dim3,void**,std::size_t,cudaStream_t);static auto real=reinterpret_cast<F>(dlsym(RTLD_NEXT,"__cudaLaunchKernel"));
 const char*name=nullptr;const auto named=cuKernelGetName(&name,reinterpret_cast<CUkernel>(kernel));
 if(named==CUDA_SUCCESS&&name&&(std::strstr(name,"coldWall")||std::strstr(name,"ColdWall")||std::strstr(name,"Cold2D")||std::strstr(name,"cold2D")))std::fprintf(stderr,"ACTUAL_COLD_LAUNCH,%s,grid=%u:%u:%u,block=%u:%u:%u,shared=%zu\n",name,grid.x,grid.y,grid.z,block.x,block.y,block.z,shared);
 return real(kernel,grid,block,args,shared,stream);
}
