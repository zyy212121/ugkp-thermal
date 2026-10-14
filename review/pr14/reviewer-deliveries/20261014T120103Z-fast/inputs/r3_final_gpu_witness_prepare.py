from r3_runner import *
source=D/'inputs/cold-actual-kernel-dimensions-S309.cpp';source.write_text(r'''#include <cuda_runtime_api.h>
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
''')
with (D/'TEST_PLAN.md').open('a') as h:h.write('\nS309补actualactivecold launch witness：S306已真实16contacted particles、perparcel/timeprofile1e-9、动态age，仅count不独立证明kernel group。外部CUDA13真实cuKernelGetName/launch ABI只打印cold kernel dims，不读取device、不改args、无额外同步；同S306原输入各CHT/FSH再运行≤300s，全部原输出bytes应相等。观察I/O不计时，helper .so源码/hash单列。codec之外host-sanitizer另用真实OpenFOAMmapper/原FSHwire解析，明确host范围不能代GPU验收。\n')
lib=ROOT/'work/fixtures/cold-actual-launch-S309.so';t=run('BUILD-COLD-ACTUAL-LAUNCH-S309',cmd('g++','-std=c++17','-O2','-shared','-fPIC','-I/usr/local/cuda/include',source,'-L/usr/local/cuda/lib64','-lcudart','-L/usr/lib/wsl/lib','-lcuda','-ldl','-o',lib),[18,20,23,26,33],900,'Pureactuallaunch metadata observer, no kernelargument/storage access.',inputs=[source],kind='BUILD',variant='DIAGNOSTIC_S309');assert t['exit_code']==0
