// Native production descriptor generation and periodic two-integer readback.
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>
#define GPU_DIRECTORY_SELECTOR directory
#define GPU_DIRECTORY_FULL 0
#define GPU_DIRECTORY_HAS_BASE_ONLY 1
#define GPU_DIRECTORY_OWNS_TILE_POLICY 0
enum class HeavyDirectoryKind:int{full=0,baseOnly=1,splitBaseAndInjection=2};
enum class CsrReductionTaskSource:int{fullIndexed=0,splitBaseDirect=1,splitInjectionIndexed=2,splitLogical=3};
struct CsrReductionTask{int cell,begin,end,source;};
struct DeviceState{
 int nCells=0,csrHeavyTileParticles=128;
 int *csrHeavyTaskCount,*csrHeavyCellCount,*csrCellTaskOffset,*csrCellTaskCount,*csrMultiTaskCellList,*cellParticleOffset,*preBaseCellOffset;
 CsrReductionTask* csrReductionTasks;
};
#include "GpuReductionTaskWrite.cuh"
#include "GpuReductionTaskCount.cuh"
#include "GpuReductionTaskMaterialize.cuh"
#include "operators/maximumDirectoryOccupancyKernel.cuh"
void ck(cudaError_t e){if(e!=cudaSuccess){std::cerr<<cudaGetErrorString(e)<<'\n';std::abort();}}
template<class T>T* array(size_t n){T*p;ck(cudaMallocManaged(&p,n*sizeof(T)));return p;}
template<class T>void deviceOwned(T*& p,size_t n){T* d;ck(cudaMalloc(&d,n*sizeof(T)));ck(cudaMemcpy(d,p,n*sizeof(T),cudaMemcpyHostToDevice));ck(cudaFree(p));p=d;}
template<class F>double timing(F f,int repeats=100){cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));ck(cudaEventRecord(a));for(int i=0;i<repeats;++i)f();ck(cudaEventRecord(b));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));return ms/repeats;}
int main(){
 for(int n:{1,1024,65536})for(bool concentrated:{false,true})for(int mode:{0,1,2}){
  DeviceState*s=array<DeviceState>(1);*s={};s->nCells=n;const int grid=(n+127)/128;
  s->cellParticleOffset=array<int>(n+1);s->preBaseCellOffset=array<int>(n+1);
  s->csrCellTaskOffset=array<int>(n+1);s->csrCellTaskCount=array<int>(n+1);
  s->csrMultiTaskCellList=array<int>(n);s->csrHeavyCellCount=array<int>(1);s->csrHeavyTaskCount=array<int>(1);
  s->cellParticleOffset[0]=s->preBaseCellOffset[0]=0;
  for(int c=0;c<n;++c){int count=concentrated?(c==n/2?1048576:0):17;
   s->cellParticleOffset[c+1]=s->cellParticleOffset[c]+count;s->preBaseCellOffset[c+1]=s->preBaseCellOffset[c]+count/2;}
  const std::vector<int> hostOffset(s->cellParticleOffset,s->cellParticleOffset+n+1),hostBase(s->preBaseCellOffset,s->preBaseCellOffset+n+1);
  const int capacity=2*n+16384;s->csrReductionTasks=array<CsrReductionTask>(capacity);
  deviceOwned(s->cellParticleOffset,n+1);deviceOwned(s->preBaseCellOffset,n+1);
  deviceOwned(s->csrCellTaskOffset,n+1);deviceOwned(s->csrCellTaskCount,n+1);
  deviceOwned(s->csrMultiTaskCellList,n);deviceOwned(s->csrHeavyCellCount,1);deviceOwned(s->csrHeavyTaskCount,1);deviceOwned(s->csrReductionTasks,capacity);
  DeviceState* device;ck(cudaMalloc(&device,sizeof(*device)));ck(cudaMemcpy(device,s,sizeof(*device),cudaMemcpyHostToDevice));
  void* scan=nullptr;size_t bytes=0;int* counts=s->csrCellTaskCount,*offsets=s->csrCellTaskOffset;
  int *heavy=s->csrHeavyCellCount,*total=s->csrHeavyTaskCount,*maximum=array<int>(1);deviceOwned(maximum,1);
  ck(cub::DeviceScan::ExclusiveSum(scan,bytes,counts,offsets,n+1));ck(cudaMalloc(&scan,bytes));
  auto count=[&]{countCsrReductionTasksKernel<<<(n+128)/128,128>>>(device,mode);};
  auto scanStep=[&]{ck(cub::DeviceScan::ExclusiveSum(scan,bytes,counts,offsets,n+1));};
  auto materialize=[&]{ck(cudaMemsetAsync(heavy,0,sizeof(int)));materializeCsrReductionTasksKernel<<<grid,128>>>(device,mode);};
  count();scanStep();materialize();ck(cudaDeviceSynchronize());
  int totalHost=0,heavyHost=0;ck(cudaMemcpy(&totalHost,total,sizeof(int),cudaMemcpyDeviceToHost));ck(cudaMemcpy(&heavyHost,heavy,sizeof(int),cudaMemcpyDeviceToHost));
  std::vector<CsrReductionTask> tasksHost(totalHost);ck(cudaMemcpy(tasksHost.data(),s->csrReductionTasks,totalHost*sizeof(CsrReductionTask),cudaMemcpyDeviceToHost));
  int expected=0,multi=0;
  for(int c=0;c<n;++c){int begin=mode==0?hostOffset[c]:mode==1?hostBase[c]:0;
   int length=mode==0?hostOffset[c+1]-begin:mode==1?hostBase[c+1]-begin:
    (hostOffset[c+1]-hostOffset[c])+(hostBase[c+1]-hostBase[c]);
   int tasks=(length+127)/128;multi+=tasks>1;
   for(int i=0;i<tasks;++i){const auto& t=tasksHost[expected++];assert(t.cell==c&&t.begin==begin+i*128&&t.end==begin+std::min(length,(i+1)*128));assert(t.source==(mode==0?0:mode==1?1:3));}
  }
  assert(expected==totalHost&&multi==heavyHost&&expected<=capacity);
  const double countMs=timing(count),scanMs=timing(scanStep),materializeMs=timing(materialize);
  auto start=std::chrono::steady_clock::now();int largest=0,tile=0;
  for(int i=0;i<100;++i){ck(cudaMemsetAsync(maximum,0,sizeof(int)));maximumDirectoryOccupancyKernel<<<grid,128>>>(device,mode,maximum);
   ck(cudaMemcpy(&largest,maximum,sizeof(int),cudaMemcpyDeviceToHost));
   ck(cudaMemcpy(&tile,reinterpret_cast<char*>(device)+offsetof(DeviceState,csrHeavyTileParticles),sizeof(int),cudaMemcpyDeviceToHost));}
  const double inspectionMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/100;
  assert(tile==128);std::cout<<"PASS cells="<<n<<" concentrated="<<concentrated<<" mode="<<mode<<" tasks="<<expected<<" max_occupancy="<<largest<<" count_ms="<<countMs<<" scan_ms="<<scanMs<<" descriptors_ms="<<materializeMs<<" inspection_ms="<<inspectionMs<<" default_amortized_inspection_ms="<<inspectionMs/100<<'\n';
  ck(cudaDeviceSynchronize());
  cudaFree(s->cellParticleOffset);cudaFree(s->preBaseCellOffset);cudaFree(s->csrCellTaskOffset);cudaFree(s->csrCellTaskCount);cudaFree(s->csrMultiTaskCellList);cudaFree(s->csrHeavyCellCount);cudaFree(s->csrHeavyTaskCount);cudaFree(s->csrReductionTasks);cudaFree(maximum);cudaFree(scan);cudaFree(device);cudaFree(s);
 }
}
