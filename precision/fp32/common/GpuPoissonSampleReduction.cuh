#pragma once
#include "CsrPersistentQueue.cuh"
#include "GpuCollisionCellCentralReduction.cuh"
// Sampling still consumes the same two Gaussian pairs per selected parcel.
// All five components form one weighted central-moment reduction.
__device__ __forceinline__ void publishPoissonSampleMoments
(DeviceState& s,int c,const GPU_OPERATOR_REAL* q) {
    s.poolThermalMass[c]=q[0];
    s.poolThermalSumUx[c]=q[1];s.poolThermalSumUy[c]=q[2];
    s.poolThermalSumUz[c]=q[3];s.poolThermalSumU2[c]=q[4];
}
__device__ __forceinline__ int poissonSampleLogicalParcel
(const DeviceState& s,int c,int pos,bool split) {
    if(!split)return s.sortedParticleIndex[pos];
    const int base=s.preBaseCellOffset[c],count=s.preBaseCellOffset[c+1]-base;
    return pos<count?base+pos:s.sortedParticleIndex[s.cellParticleOffset[c]+pos-count];
}
struct PoissonSampleTask {
    CsrReductionTask* current;GPU_OPERATOR_REAL* scratch;int thetaDrag;
    __device__ bool prepare(DeviceState& s,int task) {
        *current=s.csrReductionTasks[task];return true;
    }
    __device__ void execute(DeviceState& s,int task) {
        const auto d=*current;GPU_OPERATOR_REAL q[5]={};
        for(int pos=d.begin+threadIdx.x;pos<d.end;pos+=blockDim.x) {
            int i;
            if(d.source==static_cast<int>(CsrReductionTaskSource::splitLogical))
                i=poissonSampleLogicalParcel(s,d.cell,pos,true);
            else if(d.source==static_cast<int>(CsrReductionTaskSource::splitBaseDirect))i=pos;
            else i=s.sortedParticleIndex[pos];
            sampleOnePoissonPoolParticle<false>(s,i,thetaDrag!=0,q);
        }
        blockReduceCollisionMoments<5>(q,scratch);
        if(!threadIdx.x) {
            if(s.csrCellTaskCount[d.cell]==1)publishPoissonSampleMoments(s,d.cell,q);
            else for(int k=0;k<5;++k)s.csrCollisionPartials[8u*size_t(task)+k]=q[k];
        }
    }
};
__global__ void samplePoissonPoolTasksKernel(DeviceState* sp,int thetaDrag) {
    __shared__ CsrReductionTask current;extern __shared__ GPU_OPERATOR_REAL scratch[];
    PoissonSampleTask operation{&current,scratch,thetaDrag};
    runCsrPersistentQueue(*sp,*sp->csrHeavyTaskCount,operation);
}
struct PoissonSampleFinalize {
    GPU_OPERATOR_REAL* scratch;
    __device__ bool prepare(DeviceState&,int){return true;}
    __device__ void execute(DeviceState& s,int index) {
        const int c=s.csrMultiTaskCellList[index];GPU_OPERATOR_REAL q[5]={};
        const int first=s.csrCellTaskOffset[c],end=first+s.csrCellTaskCount[c];
        for(int task=first+threadIdx.x;task<end;task+=blockDim.x)
        {
            float other[5];
            for(int k=0;k<5;++k)other[k]=s.csrCollisionPartials[8u*size_t(task)+k];
            mergeCollisionMomentGroup(q,other);
        }
        blockReduceCollisionMoments<5>(q,scratch);
        if(!threadIdx.x)publishPoissonSampleMoments(s,c,q);
    }
};
__global__ void finalizePoissonSampleMomentsKernel(DeviceState* sp) {
    extern __shared__ GPU_OPERATOR_REAL scratch[];
    PoissonSampleFinalize operation{scratch};
    runCsrPersistentQueue(*sp,*sp->csrHeavyCellCount,operation);
}
__global__ void samplePoissonPoolCellsCentralKernel(DeviceState* sp,int directory,int thetaDrag)
{
    DeviceState& s=*sp;const int c=blockIdx.x;float q[5]={};
    const int count=collisionDirectoryCount(s,c,directory);
    for(int pos=threadIdx.x;pos<count;pos+=blockDim.x)
    {
        const int i=collisionDirectoryParticle(s,c,pos,directory);
        sampleOnePoissonPoolParticle<false>(s,i,thetaDrag!=0,q);
    }
    extern __shared__ float scratch[];
    blockReduceCollisionMoments(q,scratch);
    if(!threadIdx.x)publishPoissonSampleMoments(s,c,q);
}
#ifndef UGKP_SAMPLE_REDUCTION_DEVICE_ONLY
int launchPoissonPoolSampling(DeviceState* s,int block,int particleGrid,int directory,cudaStream_t stream) {
    const size_t sharedBytes=5u*size_t((block+31)/32)*sizeof(float);
    if(!s->csrCellLocalPathEnabled) {
        samplePoissonPoolCellsCentralKernel<<<s->nCells,block,sharedBytes,stream>>>(s->deviceState,0,0);
        const auto error=cudaGetLastError();
        if(error!=cudaSuccess){setLastError("sample Poisson particles",error);return 1;}
        return 0;
    }
    return launchCsrConditional(s,8+directory,block,directory,0,[&](cudaStream_t queue,bool heavy){
        cudaError_t error=cudaSuccess;
        if(heavy) {
            error=resetCsrPersistentQueue(s,queue);
            if(error==cudaSuccess) {
                samplePoissonPoolTasksKernel<<<s->csrHeavyWorkerGrid,block,sharedBytes,queue>>>(s->deviceState,0);
                error=cudaGetLastError();
            }
            if(error==cudaSuccess)error=resetCsrPersistentQueue(s,queue);
            if(error==cudaSuccess) {
                finalizePoissonSampleMomentsKernel<<<(s->multiprocessorCount<s->nCells?s->multiprocessorCount:s->nCells),block,sharedBytes,queue>>>(s->deviceState);
                error=cudaGetLastError();
            }
        } else {
            // Light and unsorted-fallback rows use the same cell-owned group merge.
            samplePoissonPoolCellsCentralKernel<<<s->nCells,block,sharedBytes,queue>>>(s->deviceState,directory,0);
            error=cudaGetLastError();
        }
        if(error!=cudaSuccess){setLastError("CSR Poisson sampling moments",error);return 1;}
        return 0;
    });
}
#endif
