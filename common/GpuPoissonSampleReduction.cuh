#pragma once
#include "CsrPersistentQueue.cuh"
// Sampling still consumes the same two Gaussian pairs per selected parcel.
// Only the ownership and reduction of its four weighted moments changes.
__device__ __forceinline__ void publishPoissonSampleMoments
(DeviceState& s,int c,const GPU_OPERATOR_REAL* q) {
    s.poolThermalSumUx[c]=q[0];s.poolThermalSumUy[c]=q[1];
    s.poolThermalSumUz[c]=q[2];s.poolThermalSumU2[c]=q[3];
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
        const auto d=*current;GPU_OPERATOR_REAL q[4]={};
        for(int pos=d.begin+threadIdx.x;pos<d.end;pos+=blockDim.x) {
            int i;
            if(d.source==static_cast<int>(CsrReductionTaskSource::splitLogical))
                i=poissonSampleLogicalParcel(s,d.cell,pos,true);
            else if(d.source==static_cast<int>(CsrReductionTaskSource::splitBaseDirect))i=pos;
            else i=s.sortedParticleIndex[pos];
            sampleOnePoissonPoolParticle<false>(s,i,thetaDrag!=0,q);
        }
        blockReduceComponentSums<4>(q,scratch);
        if(!threadIdx.x) {
            if(s.csrCellTaskCount[d.cell]==1)publishPoissonSampleMoments(s,d.cell,q);
            else for(int k=0;k<4;++k)s.csrHeavyPartials[8u*size_t(task)+k]=q[k];
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
        const int c=s.csrMultiTaskCellList[index];GPU_OPERATOR_REAL q[4]={};
        const int first=s.csrCellTaskOffset[c],end=first+s.csrCellTaskCount[c];
        for(int task=first+threadIdx.x;task<end;task+=blockDim.x)
            for(int k=0;k<4;++k)q[k]+=s.csrHeavyPartials[8u*size_t(task)+k];
        blockReduceComponentSums<4>(q,scratch);
        if(!threadIdx.x)publishPoissonSampleMoments(s,c,q);
    }
};
__global__ void finalizePoissonSampleMomentsKernel(DeviceState* sp) {
    extern __shared__ GPU_OPERATOR_REAL scratch[];
    PoissonSampleFinalize operation{scratch};
    runCsrPersistentQueue(*sp,*sp->csrHeavyCellCount,operation);
}
#ifndef UGKP_SAMPLE_REDUCTION_DEVICE_ONLY
int launchPoissonPoolSampling(DeviceState* s,int block,int particleGrid,int directory,cudaStream_t stream) {
    if(!s->csrCellLocalPathEnabled) {
        samplePoissonPoolParticlesKernel<<<particleGrid,s->particleBlockThreads,0,stream>>>(s->deviceState,0);
        const auto error=cudaGetLastError();
        if(error!=cudaSuccess){setLastError("sample Poisson particles",error);return 1;}
        return 0;
    }
    const size_t sharedBytes=4u*size_t((block+31)/32)*sizeof(GPU_OPERATOR_REAL);
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
            // Light CSR rows have little atomic contention. Keep the parcel
            // dispatch; the heavy-row branch above amortizes its task reduction.
            samplePoissonPoolParticlesKernel<<<particleGrid,s->particleBlockThreads,0,queue>>>(s->deviceState,0);
            error=cudaGetLastError();
        }
        if(error!=cudaSuccess){setLastError("CSR Poisson sampling moments",error);return 1;}
        return 0;
    });
}
#endif
