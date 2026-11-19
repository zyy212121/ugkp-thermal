#pragma once
#include "materials/MaterialTransportMath.H"
namespace chmt {



struct ResidentMaterialState {
    SolidQ* current=nullptr;
    SolidQ* candidate=nullptr;
    SolidQ* gasChange=nullptr;
    SolidQ* candidateGasChange=nullptr;
    const SolidQ* initial=nullptr;
    MaterialMeshView mesh{},candidateMesh{};
};
CHMT_HD inline void acceptMaterialState(ResidentMaterialState& state){
    auto* q=state.current;state.current=state.candidate;state.candidate=q;
    auto* change=state.gasChange;state.gasChange=state.candidateGasChange;state.candidateGasChange=change;
    const auto mesh=state.mesh;state.mesh=state.candidateMesh;state.candidateMesh=mesh;
}
}
