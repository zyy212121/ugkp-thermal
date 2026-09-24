#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include "GpuPrecisionTypes.H"
constexpr GpuReal OfGreat=GPU_R(1e30), OfSmall=GPU_R(1e-30);
__device__ GpuReal clampRange(GpuReal x,GpuReal a,GpuReal b){return x<a?a:x>b?b:x;}
__device__ GpuReal clampMin(GpuReal x,GpuReal a){return x<a?a:x;}
enum class CsrReductionTaskSource { fullIndexed=0,splitBaseDirect=1,splitLogical=2 };
struct CsrReductionTask{int cell,begin,end,source;};
struct DeviceState{
 DeviceState* deviceState;
 int nCells,particleCapacity,csrHeavyReductionEnabled,csrHeavyTaskCapacity,csrHeavyWorkerGrid,multiprocessorCount,lightResidentBlocksPerSm;
 GpuReal rhoMin;
 int *csrHeavyTaskCount,*csrHeavyCellCount,*csrHeavyTaskCursor,*csrCellTaskCount,*csrCellTaskOffset,*csrMultiTaskCellList,*cellParticleCount,*poolThermalCount;
 CsrReductionTask* csrReductionTasks;
 GpuReal *V,*csrHeavyPartials,*poissonPoolMass,*poissonPoolMomX,*poissonPoolMomY,*poissonPoolMomZ,*poissonPoolEnergy,*poissonPoolDiameter,*poissonPoolDiameter2,*momRhoP,*momRhoUPx,*momRhoUPy,*momRhoUPz,*momRhoEP,*momRhoPD,*momRhoHpP;
};
void setLastError(const char*,cudaError_t){}
template<int NumComponents>
__device__ void blockReduceComponentSums
(
    GpuReal (&sums)[NumComponents],
    GpuReal* warpPartials
)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warpCount = (blockDim.x + 31)/32;
    constexpr unsigned int fullWarpMask = 0xffffffffu;

                                                                             
                                                                            
                                                                        
                                                                             
                                                                               
                                                                            
    if ((blockDim.x & 31) != 0)
    {
        asm("trap;");
    }
    __syncwarp(fullWarpMask);

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        #pragma unroll
        for (int component = 0; component < NumComponents; ++component)
        {
            const GpuReal other =
                __shfl_down_sync(fullWarpMask, sums[component], offset);
            if (lane + offset < 32)
            {
                sums[component] += other;
            }
        }
    }

    if (lane == 0)
    {
        #pragma unroll
        for (int component = 0; component < NumComponents; ++component)
        {
            warpPartials[component*warpCount + warp] = sums[component];
        }
    }

    __syncthreads();

    if (warp == 0)
    {
        __syncwarp(fullWarpMask);

        #pragma unroll
        for (int component = 0; component < NumComponents; ++component)
        {
            GpuReal value =
                lane < warpCount
              ? warpPartials[component*warpCount + lane]
              : GPU_R(0.0);

            for (int offset = 16; offset > 0; offset >>= 1)
            {
                const GpuReal other =
                    __shfl_down_sync(fullWarpMask, value, offset);
                if (lane + offset < 32)
                {
                    value += other;
                }
            }

            if (lane == 0)
            {
                sums[component] = value;
            }
        }
    }
}
__device__ GpuReal granularCollisionTauFromCellDevice(DeviceState&,int c){return c%4==0?OfGreat:GPU_R(1);}
__device__ void accumulateMock(int begin,int end,int source,GpuReal* sums,GpuReal* partials){
 for(int k=0;k<8;++k)sums[k]=0;
 for(int i=begin+threadIdx.x;i<end;i+=blockDim.x)for(int k=0;k<8;++k)sums[k]+=k==7?GPU_R(1):GpuReal((i%7+1)*(k+1)+source);
 blockReduceComponentSums<8>(*reinterpret_cast<GpuReal(*)[8]>(sums),partials);
}
template<bool P> __device__ void accumulateCsrSplitLogicalPoolTask(DeviceState&,int,int a,int b,GpuReal,GpuReal* sums,GpuReal* w){accumulateMock(a,b,2,sums,w);}
template<bool P> __device__ void accumulateCsrHeavyPoolTask(DeviceState&,int,int a,int b,bool direct,GpuReal,GpuReal* sums,GpuReal* w){accumulateMock(a,b,direct?1:0,sums,w);}
template<bool GatherSurvivors = false>
__device__ void accumulateCsrHeavyMomentTask(DeviceState&,int,int a,int b,GpuReal* sums,GpuReal* w){accumulateMock(a,b,0,sums,w);}
namespace original {



template<bool PoissonMode>
__global__ void accumulateCsrSegmentedPoolTasksPersistentKernel
(
    DeviceState* sp,
    const GpuTime dt
)
{
    DeviceState& s = *sp;
    __shared__ int task;
    __shared__ int taskCount;
    __shared__ CsrReductionTask descriptor;
    __shared__ GpuReal collisionProbability;
    extern __shared__ GpuReal warpPartials[];
    for (;;)
    {
        if (threadIdx.x == 0)
        {
            taskCount = *s.csrHeavyTaskCount;
            task = atomicAdd(s.csrHeavyTaskCursor, 1);
            if (task < taskCount)
            {
                descriptor = s.csrReductionTasks[task];
                if (PoissonMode)
                {
                    const GpuReal tauColl =
                        granularCollisionTauFromCellDevice(s, descriptor.cell);
                    collisionProbability =
                        (!(tauColl < GPU_R(0.5)*OfGreat) || tauColl <= OfSmall)
                      ? GPU_R(0.0)
                      : clampRange(GPU_R(1.0) - exp(-dt/tauColl), GPU_R(0.0), GPU_R(1.0));
                }
                else collisionProbability = GPU_R(1.0);
            }
        }
        __syncthreads();
        if (task >= taskCount) return;
        const int c = descriptor.cell;
            if (PoissonMode && collisionProbability <= GPU_R(0.0))
            {
                if (threadIdx.x == 0 && s.csrCellTaskCount[c] > 1)
                {
                    #pragma unroll
                    for (int component = 0; component < 8; ++component)
                    {
                        s.csrHeavyPartials
                        [
                            8u*static_cast<size_t>(task)
                          + static_cast<size_t>(component)
                        ] = GPU_R(0.0);
                    }
                }
                __syncthreads();
                continue;
            }
            GpuReal sums[8];
            if
            (
                descriptor.source == static_cast<int>
                (
                    CsrReductionTaskSource::splitLogical
                )
            )
            {
                accumulateCsrSplitLogicalPoolTask<PoissonMode>
                (
                    s, c, descriptor.begin, descriptor.end,
                    collisionProbability, sums, warpPartials
                );
            }
            else
            {
                const bool directParticleIndex =
                    descriptor.source == static_cast<int>
                    (
                        CsrReductionTaskSource::splitBaseDirect
                    );
                accumulateCsrHeavyPoolTask<PoissonMode>
                (
                    s, c, descriptor.begin, descriptor.end,
                    directParticleIndex, collisionProbability,
                    sums, warpPartials
                );
            }
            if (threadIdx.x == 0)
            {
                if (s.csrCellTaskCount[c] == 1)
                {
                    s.poissonPoolMass[c] = sums[0];
                    s.poissonPoolMomX[c] = sums[1];
                    s.poissonPoolMomY[c] = sums[2];
                    s.poissonPoolMomZ[c] = sums[3];
                    s.poissonPoolEnergy[c] = sums[4];
                    s.poissonPoolDiameter[c] = sums[5];
                    s.poissonPoolDiameter2[c] = sums[6];
                    s.poolThermalCount[c] = static_cast<int>(sums[7]);
                }
                else
                {
                    #pragma unroll
                    for (int component = 0; component < 8; ++component)
                    {
                        s.csrHeavyPartials
                        [
                            8u*static_cast<size_t>(task)
                          + static_cast<size_t>(component)
                        ] = sums[component];
                    }
                }
            }
            __syncthreads();
    }
}

__global__ void finalizeCsrSegmentedPoolCellsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    __shared__ int multiIndex;
    extern __shared__ GpuReal warpPartials[];
    for (;;)
    {
        if (threadIdx.x == 0) multiIndex = atomicAdd(s.csrHeavyTaskCursor, 1);
        __syncthreads();
        if (multiIndex >= *s.csrHeavyCellCount) return;
        const int c = s.csrMultiTaskCellList[multiIndex];
        if (!(s.csrCellTaskCount[c] > 1)) asm("trap;");
        GpuReal sums[8] = {GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0)};
        const int firstTask = s.csrCellTaskOffset[c];
        const int endTask = s.csrCellTaskOffset[c + 1];
        for (int task = firstTask + threadIdx.x; task < endTask; task += blockDim.x)
        {
            #pragma unroll
            for (int component = 0; component < 8; ++component)
            {
                sums[component] += s.csrHeavyPartials
                [
                    8u*static_cast<size_t>(task)
                  + static_cast<size_t>(component)
                ];
            }
        }
        blockReduceComponentSums<8>(sums, warpPartials);
        if (threadIdx.x == 0)
        {
            s.poissonPoolMass[c] = sums[0];
            s.poissonPoolMomX[c] = sums[1];
            s.poissonPoolMomY[c] = sums[2];
            s.poissonPoolMomZ[c] = sums[3];
            s.poissonPoolEnergy[c] = sums[4];
            s.poissonPoolDiameter[c] = sums[5];
            s.poissonPoolDiameter2[c] = sums[6];
            s.poolThermalCount[c] = static_cast<int>(sums[7]);
        }
        __syncthreads();
    }
}

int launchCsrSegmentedPoolReduction
(
    DeviceState* s,
    const GpuTime dt,
    const bool poissonMode,
    const int block
)
{
    if (s->csrHeavyReductionEnabled == 0 || s->particleCapacity <= 0) return 0;
    cudaError_t err = cudaMemset(s->csrHeavyTaskCursor, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("reset CSR segmented pool task cursor", err);
        return 1;
    }
    const int warpCount = (block + 31)/32;
    const size_t sharedBytes = 8u*static_cast<size_t>(warpCount)*sizeof(GpuReal);
    if (poissonMode)
    {
        accumulateCsrSegmentedPoolTasksPersistentKernel<true>
            <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
    }
    else
    {
        accumulateCsrSegmentedPoolTasksPersistentKernel<false>
            <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState, dt);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("CSR segmented pool worker launch", err);
        return 1;
    }
    err = cudaMemset(s->csrHeavyTaskCursor, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("reset CSR segmented pool finalize cursor", err);
        return 1;
    }
    const int finalizeGrid =
        s->multiprocessorCount < s->nCells ? s->multiprocessorCount : s->nCells;
    finalizeCsrSegmentedPoolCellsKernel
        <<<finalizeGrid, block, sharedBytes>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("finalizeCsrSegmentedPoolCellsKernel launch", err);
        return 1;
    }
    return 0;
}




__global__ void accumulateCsrSegmentedMomentTasksPersistentKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    __shared__ int task;
    __shared__ int taskCount;
    __shared__ CsrReductionTask descriptor;
    extern __shared__ GpuReal warpPartials[];
    for (;;)
    {
        if (threadIdx.x == 0)
        {
            taskCount = *s.csrHeavyTaskCount;
            task = atomicAdd(s.csrHeavyTaskCursor, 1);
            if (task < taskCount)
            {
                descriptor = s.csrReductionTasks[task];
            }
        }
        __syncthreads();
        if (task >= taskCount) return;
        const int c = descriptor.cell;
            GpuReal sums[8];
            accumulateCsrHeavyMomentTask
            (
                s, c, descriptor.begin, descriptor.end, sums, warpPartials
            );
            if (threadIdx.x == 0)
            {
                if (s.csrCellTaskCount[c] == 1)
                {
                    s.cellParticleCount[c] = static_cast<int>(sums[7]);
                    if (c == 0) s.cellParticleCount[s.nCells] = 0;
                    const GpuReal invV = GPU_R(1.0)/clampMin(s.V[c], s.rhoMin);
                    s.momRhoP[c] = sums[0]*invV;
                    s.momRhoUPx[c] = sums[1]*invV;
                    s.momRhoUPy[c] = sums[2]*invV;
                    s.momRhoUPz[c] = sums[3]*invV;
                    s.momRhoEP[c] = sums[4]*invV;
                    s.momRhoPD[c] = sums[5]*invV;
                    s.momRhoHpP[c] = sums[6]*invV;
                }
                else
                {
                    #pragma unroll
                    for (int component = 0; component < 8; ++component)
                    {
                        s.csrHeavyPartials
                        [
                            8u*static_cast<size_t>(task)
                          + static_cast<size_t>(component)
                        ] = sums[component];
                    }
                }
            }
            __syncthreads();
    }
}

__global__ void finalizeCsrSegmentedMomentCellsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    __shared__ int multiIndex;
    extern __shared__ GpuReal warpPartials[];
    for (;;)
    {
        if (threadIdx.x == 0) multiIndex = atomicAdd(s.csrHeavyTaskCursor, 1);
        __syncthreads();
        if (multiIndex >= *s.csrHeavyCellCount) return;
        const int c = s.csrMultiTaskCellList[multiIndex];
        if (!(s.csrCellTaskCount[c] > 1)) asm("trap;");
        GpuReal sums[8] = {GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0)};
        const int firstTask = s.csrCellTaskOffset[c];
        const int endTask = s.csrCellTaskOffset[c + 1];
        for (int task = firstTask + threadIdx.x; task < endTask; task += blockDim.x)
        {
            #pragma unroll
            for (int component = 0; component < 8; ++component)
            {
                sums[component] += s.csrHeavyPartials
                [
                    8u*static_cast<size_t>(task)
                  + static_cast<size_t>(component)
                ];
            }
        }
        blockReduceComponentSums<8>(sums, warpPartials);
        if (threadIdx.x == 0)
        {
            s.cellParticleCount[c] = static_cast<int>(sums[7]);
            if (c == 0) s.cellParticleCount[s.nCells] = 0;
            const GpuReal invV = GPU_R(1.0)/clampMin(s.V[c], s.rhoMin);
            s.momRhoP[c] = sums[0]*invV;
            s.momRhoUPx[c] = sums[1]*invV;
            s.momRhoUPy[c] = sums[2]*invV;
            s.momRhoUPz[c] = sums[3]*invV;
            s.momRhoEP[c] = sums[4]*invV;
            s.momRhoPD[c] = sums[5]*invV;
            s.momRhoHpP[c] = sums[6]*invV;
        }
        __syncthreads();
    }
}

int launchCsrSegmentedMomentReduction(DeviceState* s, const int block)
{
    if (s->csrHeavyReductionEnabled == 0 || s->particleCapacity <= 0) return 0;
    cudaError_t err = cudaMemset(s->csrHeavyTaskCursor, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("reset CSR segmented moment task cursor", err);
        return 1;
    }
    const int warpCount = (block + 31)/32;
    const size_t sharedBytes = 8u*static_cast<size_t>(warpCount)*sizeof(GpuReal);
    accumulateCsrSegmentedMomentTasksPersistentKernel
        <<<s->csrHeavyWorkerGrid, block, sharedBytes>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("CSR segmented moment worker launch", err);
        return 1;
    }
    err = cudaMemset(s->csrHeavyTaskCursor, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        setLastError("reset CSR segmented moment finalize cursor", err);
        return 1;
    }
    const int finalizeGrid =
        s->multiprocessorCount < s->nCells ? s->multiprocessorCount : s->nCells;
    finalizeCsrSegmentedMomentCellsKernel
        <<<finalizeGrid, block, sharedBytes>>>(s->deviceState);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        setLastError("finalizeCsrSegmentedMomentCellsKernel launch", err);
        return 1;
    }
    return 0;
}

}
#include "CsrSegmentedPoolWorkers.cuh"
#include "CsrSegmentedMomentWorkers.cuh"
template<class T> void alloc(T*& p,int n){if(cudaMallocManaged(&p,std::max(n,1)*sizeof(T))!=cudaSuccess)exit(2);}
void checkCuda(){auto e=cudaDeviceSynchronize();if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
int main(){
 for(int cells:{1,3,257})for(int sm:{2,128})for(int block:{32,64,128,256}){
  DeviceState* s;alloc(s,1);s->deviceState=s;s->nCells=cells;s->multiprocessorCount=sm;s->lightResidentBlocksPerSm=2;s->csrHeavyWorkerGrid=sm;s->csrHeavyReductionEnabled=1;s->rhoMin=GPU_R(1e-12);
  std::vector<int> counts(cells),offset(cells+1,0);for(int c=0;c<cells;++c){counts[c]=c%11==0?8193:c%5==0?0:c%65;offset[c+1]=offset[c]+counts[c];}
  int population=offset.back();s->particleCapacity=population+1;s->csrHeavyTaskCapacity=population/block+2*cells+2;
  int shares=(std::max(population,1)+block*sm*2-1)/(block*sm*2);int tile=block*shares;
  alloc(s->csrHeavyTaskCount,1);alloc(s->csrHeavyCellCount,1);alloc(s->csrHeavyTaskCursor,1);alloc(s->csrCellTaskCount,cells);alloc(s->csrCellTaskOffset,cells+1);alloc(s->csrMultiTaskCellList,cells);alloc(s->csrReductionTasks,s->csrHeavyTaskCapacity);alloc(s->csrHeavyPartials,8*s->csrHeavyTaskCapacity);alloc(s->V,cells);alloc(s->cellParticleCount,cells+1);alloc(s->poolThermalCount,cells);
  GpuReal** fields[]={&s->poissonPoolMass,&s->poissonPoolMomX,&s->poissonPoolMomY,&s->poissonPoolMomZ,&s->poissonPoolEnergy,&s->poissonPoolDiameter,&s->poissonPoolDiameter2,&s->momRhoP,&s->momRhoUPx,&s->momRhoUPy,&s->momRhoUPz,&s->momRhoEP,&s->momRhoPD,&s->momRhoHpP};
  for(auto pp:fields)alloc(*pp,cells);
  int tasks=0,multi=0;for(int c=0;c<cells;++c){s->V[c]=1;s->csrCellTaskOffset[c]=tasks;int count=counts[c]==0?0:(counts[c]+tile-1)/tile;s->csrCellTaskCount[c]=count;if(count>1)s->csrMultiTaskCellList[multi++]=c;for(int k=0;k<count;++k)s->csrReductionTasks[tasks++]={c,offset[c]+k*tile,std::min(offset[c]+(k+1)*tile,offset[c+1]),c%3};}
  s->csrCellTaskOffset[cells]=tasks;*s->csrHeavyTaskCount=tasks;*s->csrHeavyCellCount=multi;
  DeviceState host=*s;std::vector<GpuReal*> values;for(auto pp:fields)values.push_back(*pp);
  for(bool poisson:{false,true})for(bool useOriginal:{true,false}){
   for(auto ptr:values)cudaMemset(ptr,0,cells*sizeof(GpuReal));cudaMemset(host.cellParticleCount,0,(cells+1)*sizeof(int));cudaMemset(host.poolThermalCount,0,cells*sizeof(int));
   if(useOriginal){original::launchCsrSegmentedPoolReduction(&host,.01,poisson,block);original::launchCsrSegmentedMomentReduction(&host,block);}
   else{*s->csrHeavyTaskCursor=123456;launchCsrSegmentedPoolReduction(&host,.01,poisson,block);launchCsrSegmentedMomentReduction(&host,block);}checkCuda();
   for(int c=0;c<cells;++c){for(int k=0;k<14;++k){double want=0;bool emptyPool=k<7 && poisson && c%4==0;if(!emptyPool)for(int i=offset[c];i<offset[c+1];++i)want+=(i%7+1)*(k%7+1)+(k<7?c%3:0);if((*fields[k])[c]!=GpuReal(want)){fprintf(stderr,"wrong output cell=%d component=%d old=%d\n",c,k,useOriginal);return 4;}}
    if(s->cellParticleCount[c]!=counts[c]||s->poolThermalCount[c]!=(poisson&&c%4==0?0:counts[c]))return 5;
   }
  }
  for(auto pp:fields)cudaFree(*pp);cudaFree(s->csrHeavyTaskCount);cudaFree(s->csrHeavyCellCount);cudaFree(s->csrHeavyTaskCursor);cudaFree(s->csrCellTaskCount);cudaFree(s->csrCellTaskOffset);cudaFree(s->csrMultiTaskCellList);cudaFree(s->csrReductionTasks);cudaFree(s->csrHeavyPartials);cudaFree(s->V);cudaFree(s->cellParticleCount);cudaFree(s->poolThermalCount);cudaFree(s);
 }
 puts("shared/legacy persistent dispatch agree with independent sums, both modes, all sources, empty/multi cells and reused queues");
}
