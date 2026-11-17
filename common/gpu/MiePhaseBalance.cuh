#ifndef UGKP_GPU_MIE_PHASE_BALANCE_CUH
#define UGKP_GPU_MIE_PHASE_BALANCE_CUH
#include "MieOpticalProperties.H"
#include <cuda_runtime.h>
namespace ugkp { namespace gpu { namespace mieDetail {
__device__ inline RadiationReal orderedMaximum(RadiationReal a,RadiationReal b){return a>b?a:b;}
__device__ inline RadiationReal phaseProbability(MieOpticalTableView table,const MieBracket& diameter,
    const MieBracket& temperature,const RadiationReal* weights,const MieBracket* cosines,
    const RadiationReal* scaling,int directions,int row,int column){
    const RadiationReal raw=miePhaseAt(table,diameter,temperature,cosines[row*directions+column])
        /RadiationReal(12.56637061435917295385057353311801153679);
    return weights[column]*scaling[row]*raw*scaling[column];
}
static __global__ void phaseBrackets(int count,MieOpticalTableView table,const RadiationReal* cosines,MieBracket* brackets){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)brackets[i]=mieLinearBracket(table.mu,table.angles,cosines[i]);
}
static __global__ void balancePhase(int count,int directions,MieOpticalTableView table,const MieOpticalQuery* queries,
    const RadiationReal* weights,const MieBracket* cosines,RadiationReal tolerance,int maximumIterations,
    RadiationReal* scaling,RadiationReal* nextScaling,RadiationReal* rowError,RadiationReal* columnError,MiePhaseMetrics* metrics,
    const RadiationReal* scattering=nullptr){
    const int cell=blockIdx.x;if(cell>=count)return;
    const std::size_t offset=std::size_t(cell)*directions;
    RadiationReal* current=scaling+offset;RadiationReal* next=nextScaling+offset;
    RadiationReal* rows=rowError+offset;RadiationReal* columns=columnError+offset;
    if(scattering&&scattering[cell]==0){
        for(int row=threadIdx.x;row<directions;row+=blockDim.x)current[row]=1;
        if(!threadIdx.x)metrics[cell]={};return;
    }
    const auto diameter=mieLogBracket(table.diameter,table.diameters,queries[cell].diameter);
    const auto temperature=mieLinearBracket(table.temperature,table.temperatures,queries[cell].temperature);
    for(int row=threadIdx.x;row<directions;row+=blockDim.x)current[row]=1;
    __shared__ bool finished;
    __syncthreads();
    const int limit=maximumIterations>1?maximumIterations:1;
    for(int iteration=1;iteration<=limit;++iteration){
        for(int row=threadIdx.x;row<directions;row+=blockDim.x){
            RadiationReal sum=0;for(int column=0;column<directions;++column)
                sum+=phaseProbability(table,diameter,temperature,weights,cosines,current,directions,row,column);
            next[row]=current[row]/sqrt(sum);
        }
        __syncthreads();
        for(int row=threadIdx.x;row<directions;row+=blockDim.x)current[row]=next[row];
        __syncthreads();
        for(int row=threadIdx.x;row<directions;row+=blockDim.x){
            RadiationReal sum=0,weightedColumn=0;
            for(int column=0;column<directions;++column){
                sum+=phaseProbability(table,diameter,temperature,weights,cosines,current,directions,row,column);
                weightedColumn+=weights[column]*phaseProbability(table,diameter,temperature,weights,cosines,current,directions,column,row);
            }
            rows[row]=fabs(sum-1);columns[row]=fabs(weightedColumn-weights[row])/fabs(weights[row]);
        }
        __syncthreads();
        if(!threadIdx.x){
            MiePhaseMetrics result;
            // Preserve ordered max (including IEEE NaNs), not a reordered reduction.
            for(int row=0;row<directions;++row){
                result.rowError=orderedMaximum(result.rowError,rows[row]);
                result.columnError=orderedMaximum(result.columnError,columns[row]);
                result.scalingCorrection=orderedMaximum(result.scalingCorrection,fabs(current[row]-1));
            }
            result.iterations=iteration;metrics[cell]=result;
            finished=result.rowError<=tolerance&&result.columnError<=tolerance;
        }
        __syncthreads();if(finished)break;
    }
}

}}}
#endif
