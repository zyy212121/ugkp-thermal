#pragma once
#include "materials/MaterialReactionMath.H"
#include "materials/MaterialInventoryMath.H"
namespace chmt { namespace materialSourceDetail {
struct MaterialStepResult {int phase=0,code=0,entity=-1;bool numerical=true;};
__device__ MaterialStepResult firstMaterialFailure(MaterialStepResult a,MaterialStepResult b,bool phasePriority){
    if(!a.phase)return b;if(!b.phase)return a;
    if(phasePriority&&a.phase!=b.phase)return a.phase<b.phase?a:b;
    return a.entity<=b.entity?a:b;
}
__device__ void reduceMaterialFailure(MaterialStepResult value,MaterialStepResult* result,bool phasePriority){
    extern __shared__ MaterialStepResult failureValues[];
    auto* values=failureValues;
    const int t=threadIdx.x;values[t]=value;__syncthreads();
    for(int active=blockDim.x;active>1;active=(active+1)/2){const int stride=(active+1)/2;
        if(t<active-stride)values[t]=firstMaterialFailure(values[t],values[t+stride],phasePriority);
        __syncthreads();
    }
    if(t==0)*result=values[0];
}
__device__ bool finishMaterialSourceCell(int c,SolidQ& result,
    const int* radiationOffsets,const int* radiationFaces,const Real* radiation,
    const unsigned* receivers,const Real* volumes,const PhysicsConfig* physics,
    Real dt,bool finalStep,MaterialStepResult* status,
    Real reactionMaxStep,Real reactionCfl,int reactionBudget){
    if(radiation)for(int e=radiationOffsets[c];e<radiationOffsets[c+1];++e){
        const int face=radiationFaces[e];
        if(receivers[face]==ConsumeSolid)result.energy+=radiation[face];
    }



    if(!validSolidInventory(result)){status[c]={3,0,c,true};return false;}
    SolidQ reactedResult;
    const auto reaction=advanceMaterialReactionsMath(result,volumes[c],*physics,
        dt,reactionMaxStep,reactionCfl,reactionBudget,reactedResult);
    if(reaction.code!=MaterialReactionCode::Success){status[c]={1,int(reaction.code),c,reaction.numericalFailure};return false;}
    result=reactedResult;
    MaterialPrimitive primitive;bool numerical=true;
    if(!validSolidInventory(result)||(!finalStep&&!recoverMaterial(result,volumes[c],*physics,primitive,nullptr,-1,&numerical))){
        status[c]={3,0,c,numerical};return false;
    }
    if(!finalStep)result.porosity=primitive.porosity;
    return true;
}
template<class Packets>
__device__ void applyMaterialSourceCell(int c,const SolidQ* current,const SolidQ* rates,
    const SolidQ* initial,const SolidQ* gasChange,SolidQ* candidateGasChange,Packets packets,const int* offsets,
    const int* indices,const int* radiationOffsets,const int* radiationFaces,
    const Real* radiation,const unsigned* receivers,const Real* volumes,const PhysicsConfig* physics,
    Real dt,bool finalStep,SolidQ* next,SolidQ* donor,MaterialStepResult* status,
    Real reactionMaxStep,Real reactionCfl,int reactionBudget){
    status[c]={};
    const SolidQ transported=addSolid(current[c],rates[c],dt);
    SolidQ result;
    const auto code=applyMaterialPacketRowView(transported,packets,indices,offsets[c],offsets[c+1],result);
    if(code!=MaterialInventoryCode::Success){status[c]={2,int(code),c,true};return;}
    if(!finishMaterialSourceCell(c,result,radiationOffsets,radiationFaces,radiation,
        receivers,volumes,physics,dt,finalStep,status,reactionMaxStep,reactionCfl,reactionBudget))return;
    SolidQ change=gasChange[c];
    for(int e=offsets[c];e<offsets[c+1];++e)
        change=addSolid(change,packetDelta(packets[indices[e]]).solid);
    candidateGasChange[c]=change;next[c]=result;
    if(donor)donor[c]=addSolid(addSolid(result,initial[c],-1),change,-1);
}
__global__ void applyMaterialSourcesKernel(int count,const SolidQ* current,const SolidQ* rates,
    const SolidQ* initial,const SolidQ* gasChange,SolidQ* candidateGasChange,const ExchangePacket* packets,const int* offsets,
    const int* indices,const int* radiationOffsets,const int* radiationFaces,
    const Real* radiation,const unsigned* receivers,const Real* volumes,const PhysicsConfig* physics,
    Real dt,bool finalStep,SolidQ* next,SolidQ* donor,MaterialStepResult* status,
    Real reactionMaxStep,Real reactionCfl,int reactionBudget){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=count)return;
    applyMaterialSourceCell(c,current,rates,initial,gasChange,candidateGasChange,packets,offsets,
        indices,radiationOffsets,radiationFaces,radiation,receivers,volumes,physics,dt,finalStep,
        next,donor,status,reactionMaxStep,reactionCfl,reactionBudget);
}


__global__ void beginStreamedMaterialPackets(int count,const SolidQ* current,
    const SolidQ* rates,const SolidQ* gasChange,Real dt,SolidQ* transported,
    SolidQ* candidateGasChange,MaterialPacketAccumulator* accumulated,MaterialStepResult* status){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=count)return;
    status[c]={};transported[c]=addSolid(current[c],rates[c],dt);
    accumulated[c].begin(transported[c]);candidateGasChange[c]=gasChange[c];
}
__global__ void appendStreamedMaterialPackets(int count,const ExchangePacket* packets,
    const int* offsets,const int* indices,MaterialPacketAccumulator* accumulated,SolidQ* candidateGasChange){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=count)return;
    auto row=accumulated[c];auto change=candidateGasChange[c];
    for(int e=offsets[c];e<offsets[c+1];++e){
        const auto packet=packets[indices[e]];row.append(packet);
        change=addSolid(change,packetDelta(packet).solid);
    }
    accumulated[c]=row;candidateGasChange[c]=change;
}
__global__ void finishStreamedMaterialSources(int count,const MaterialPacketAccumulator* accumulated,
    const SolidQ* initial,const SolidQ* candidateGasChange,
    const int* radiationOffsets,const int* radiationFaces,const Real* radiation,
    const unsigned* receivers,const Real* volumes,const PhysicsConfig* physics,
    Real dt,bool finalStep,SolidQ* next,SolidQ* donor,MaterialStepResult* status,
    Real reactionMaxStep,Real reactionCfl,int reactionBudget){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=count)return;
    SolidQ result;const auto code=accumulated[c].finish(next[c],result);
    if(code!=MaterialInventoryCode::Success){status[c]={2,int(code),c,true};return;}
    if(!finishMaterialSourceCell(c,result,radiationOffsets,radiationFaces,radiation,
        receivers,volumes,physics,dt,finalStep,status,reactionMaxStep,reactionCfl,reactionBudget))return;
    next[c]=result;
    if(donor)donor[c]=addSolid(addSolid(result,initial[c],-1),candidateGasChange[c],-1);
}
__global__ void selectSourceFailure(const MaterialStepResult* status,int count,MaterialStepResult* result){
    if(result->phase)return;
    MaterialStepResult value{};
    for(int c=threadIdx.x;c<count;c+=blockDim.x)
        value=firstMaterialFailure(value,status[c],true);
    reduceMaterialFailure(value,result,true);
}
}}
