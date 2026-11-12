#pragma once
#include "gpu/ResidentMaterialState.H"
#include "gpu/ResidentMaterialSlice.H"
#include "gpu/ResidentMaterialStep.H"
#include "gpu/MaterialSourceKernels.cuh"
namespace chmt {
// The enclosing material graph admits the step only after the driving,
// transport, surface and geometry phases succeed. The row/source mathematics
// is identical to the contiguous-packet entry point; only storage is borrowed.
__global__ void applyResidentMaterialSources(const ResidentMaterialState* state,
    const ResidentMaterialStep* step,const SolidQ* rates,ResidentMaterialSlice packets,
    const int* offsets,const int* indices,const int* radiationOffsets,const int* radiationFaces,
    const EnergyReal* radiation,const unsigned* receivers,const PhysicsConfig* physics,
    SolidQ* donor,materialSourceDetail::MaterialStepResult* status,
    Real reactionMaxStep,PhysicalReal reactionCfl,int reactionBudget){
    if(!step->active||step->error!=MaterialStepSelectionError::None||packets.plan->error!=ugkp::IntervalSliceError::None)return;
    const auto binding=*state;const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=binding.mesh.cells)return;
    materialSourceDetail::applyMaterialSourceCell(c,binding.current,rates,binding.initial,
        binding.gasChange,binding.candidateGasChange,packets,offsets,indices,radiationOffsets,radiationFaces,
        radiation,receivers,binding.candidateMesh.volumes,physics,step->dt,step->finalStep,
        binding.candidate,donor,status,reactionMaxStep,reactionCfl,reactionBudget);
}
// This flag belongs to the whole material transaction, including endpoint
// surface evaluation, not just its local source solve. Failed candidates keep
// all accepted inventory, accumulated gas transfer and geometry bindings.
__global__ void commitResidentMaterialState(ResidentMaterialState* state,const int* accepted){
    if(!blockIdx.x&&!threadIdx.x&&*accepted)acceptMaterialState(*state);
}
}
