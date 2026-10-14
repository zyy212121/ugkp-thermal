#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
extern "C" cudaError_t cudaLaunchKernel(const void* function,dim3 grid,dim3 block,void**args,std::size_t shared,cudaStream_t stream){
 using F=cudaError_t(*)(const void*,dim3,dim3,void**,std::size_t,cudaStream_t);static auto real=reinterpret_cast<F>(dlsym(RTLD_NEXT,"cudaLaunchKernel"));static unsigned long long calls=0;
 const char*name=nullptr;const auto named=cudaFuncGetName(&name,function);
 if(named==cudaSuccess&&name&&std::strstr(name,"advanceGasChemistryKernel")){
  const double* volume=*reinterpret_cast<const double**>(args[2]);double values[8]{};const auto copied=cudaMemcpy(values,volume,sizeof(values),cudaMemcpyDeviceToHost);
  std::fprintf(stderr,"CHEM_VOLUME,%llu,%a,%d,%p,%d",++calls,*reinterpret_cast<double*>(args[1]),int(*reinterpret_cast<bool*>(args[3])),static_cast<const void*>(volume),int(copied));for(double v:values)std::fprintf(stderr,",%a",v);std::fprintf(stderr,"\n");
 }
 return real(function,grid,block,args,shared,stream);
}
