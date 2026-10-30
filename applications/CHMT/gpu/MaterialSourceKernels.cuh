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
    __shared__ MaterialStepResult values[256];
    const int t=threadIdx.x;values[t]=value;__syncthreads();
    for(int stride=blockDim.x/2;stride;stride/=2){
        if(t<stride)values[t]=firstMaterialFailure(values[t],values[t+stride],phasePriority);
        __syncthreads();
    }
    if(t==0)*result=values[0];
}
__global__ void applyMaterialSourcesKernel(int count,const SolidQ* current,const SolidQ* rates,
    const SolidQ* initial,const SolidQ* gasChange,SolidQ* candidateGasChange,const ExchangePacket* packets,const int* offsets,
    const int* indices,const int* radiationOffsets,const int* radiationFaces,
    const Real* radiation,const unsigned* receivers,const Real* volumes,const PhysicsConfig* physics,
    Real dt,bool finalStep,SolidQ* next,SolidQ* donor,MaterialStepResult* status,
    Real reactionMaxStep,Real reactionCfl,int reactionBudget){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=count)return;
    status[c]={};
    const SolidQ transported=addSolid(current[c],rates[c],dt);
    SolidQ result;
    const auto code=applyMaterialPacketRow(transported,packets,indices,offsets[c],offsets[c+1],result);
    if(code!=MaterialInventoryCode::Success){status[c]={2,int(code),c,true};return;}
    if(radiation)for(int e=radiationOffsets[c];e<radiationOffsets[c+1];++e){
        const int face=radiationFaces[e];
        if(receivers[face]==ConsumeSolid)result.energy+=radiation[face];
    }
    // Lie splitting uses a single consistent inventory: transport and accepted
    // boundary packets first, then the unchanged positive reaction integrator.
    // Reaction cannot consume inventory already spent by old-state transport.
    if(!validSolidInventory(result)){status[c]={3,0,c,true};return;}
    SolidQ reactedResult;
    const auto reaction=advanceMaterialReactionsMath(result,volumes[c],*physics,
        dt,reactionMaxStep,reactionCfl,reactionBudget,reactedResult);
    if(reaction.code!=MaterialReactionCode::Success){status[c]={1,int(reaction.code),c,reaction.numericalFailure};return;}
    result=reactedResult;
    MaterialPrimitive primitive;bool numerical=true;
    if(!validSolidInventory(result)||(!finalStep&&!recoverMaterial(result,volumes[c],*physics,primitive,nullptr,-1,&numerical))){
        status[c]={3,0,c,numerical};return;
    }
    if(!finalStep)result.porosity=primitive.porosity;
    SolidQ change=gasChange[c];
    for(int e=offsets[c];e<offsets[c+1];++e)
        change=addSolid(change,packetDelta(packets[indices[e]]).solid);
    candidateGasChange[c]=change;next[c]=result;
    if(donor)donor[c]=addSolid(addSolid(result,initial[c],-1),change,-1);
}
__global__ void selectSourceFailure(const MaterialStepResult* status,int count,MaterialStepResult* result){
    if(result->phase)return; // block-uniform; reaction failure retains precedence
    MaterialStepResult value{};
    for(int c=threadIdx.x;c<count;c+=blockDim.x)
        value=firstMaterialFailure(value,status[c],true);
    reduceMaterialFailure(value,result,true);
}
}}
